use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::slice;
use std::error::Error;
use std::mem::size_of;

use ash::vk::{CommandBuffer, Buffer, BufferUsageFlags, MemoryMapFlags, ShaderStageFlags, DeviceSize, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, DescriptorSetAllocateInfo, DescriptorSet, WriteDescriptorSet, DescriptorBufferInfo, PipelineBindPoint, IndexType};
use nalgebra::{Rotation3, Scale3, Translation3, Vector3, Matrix4};
use obj::{load_obj, Obj, TexturedVertex};

use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;
use crate::vulkan::vulkan_allocator::Allocation;

pub struct Planet {
    texture: Texture,
    radius: f32, // size of sphere
    orbit: f32, // dist from sun
    year: f32, // time to orbit sun
    day: f32, // time to revolve around self
    phase: f32 // initial phase in orbit
}

impl Planet {
    pub fn position(&self, time: f32) -> [f32; 3] {
        [
            ((2.0 * PI * time) / self.year + self.phase).cos() * self.orbit,
            0.0,
            ((2.0 * PI * time) / self.year + self.phase).sin() * self.orbit
        ]
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, vp: &[f32], index_count: u32, time: f32) {
        unsafe {
            // Bind planet texture
            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.texture.descriptor_set], &[]);

            let model = Translation3::from(self.position(time)).to_homogeneous()
                * Rotation3::from_axis_angle(&Vector3::y_axis(), 2.0 * PI * time / self.day).to_homogeneous()
                * Scale3::from([self.radius, self.radius, self.radius]).to_homogeneous();
            let data = [model.as_slice(), vp].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_draw_indexed(cmdbuf, index_count, 1, 0, 0, 0);
        }
    }
}

pub struct System {
    planets: Vec<Planet>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
    stars: Texture,
    sun: Texture,
    uniform_allocations: Vec<Allocation>,
    descriptor_sets: Vec<DescriptorSet>
}

impl System {
    pub fn new(vulkan: &VulkanInstance) -> Result<System, Box<dyn Error>> {
        let planets = vec![
            Planet {
                texture: Texture::new(vulkan, "res/2k_mercury.jpg")?,
                radius: 0.2,
                orbit: 2.0,
                year: 100.0,
                day: 95.0,
                phase: 1.3 * PI
            },
            Planet {
                texture: Texture::new(vulkan, "res/2k_venus_surface.jpg")?,
                radius: 0.35,
                orbit: 4.5,
                year: 150.0,
                day: 25.0,
                phase: 0.7 * PI
            },
            Planet {
                texture: Texture::new(vulkan, "res/2k_earth_daymap.jpg")?,
                radius: 0.5,
                orbit: 7.8,
                year: 250.0,
                day: 40.0,
                phase: 0.1 * PI
            },
            Planet {
                texture: Texture::new(vulkan, "res/2k_earth_clouds.png")?,
                radius: 0.54,
                orbit: 7.8,
                year: 250.0,
                day: 50.0,
                phase: 0.1 * PI
            },
            Planet {
                texture: Texture::new(vulkan, "res/2k_mars.jpg")?,
                radius: 0.45,
                orbit: 9.0,
                year: 300.0,
                day: 48.0,
                phase: 1.7 * PI
            }
        ];

        // Load textures
        let [stars, sun] = ["res/2k_stars_milky_way.jpg", "res/2k_sun.jpg"].map(|file| Texture::new(vulkan, file));

        // Load model
        let file = BufReader::new(File::open("res/sphere.obj")?);
        let sphere: Obj<TexturedVertex> = load_obj(file)?;

        // Allocate & write vertex buffer
        let vertices = sphere.vertices.iter().map(|v| {
            let mut vec = [v.position, v.normal].concat();
            vec.extend_from_slice(&v.texture[..2]);
            vec
        }).flatten().collect::<Vec<_>>();
        let (vertex_buffer, _vertex_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::VERTEX_BUFFER, (vertices.len() * 4) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut f32).copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        // Allocate & write index buffer
        let (index_buffer, _index_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::INDEX_BUFFER, (sphere.indices.len() * 2) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut u16).copy_from_nonoverlapping(sphere.indices.as_ptr(), sphere.indices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        // Allocate uniform buffers
        let (uniform_buffers, uniform_allocations) = vulkan.allocator.allocate_buffer_chain(
            &vulkan.device, BufferUsageFlags::UNIFORM_BUFFER, size_of::<f32>() as DeviceSize, vulkan.swapchain.len())?;

        // Construct descriptor sets to bind uniform buffers
        let descriptor_pool = unsafe {
            let pool_sizes = [DescriptorPoolSize {
                ty: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: vulkan.swapchain.len() as u32
            }];
            let create_info = DescriptorPoolCreateInfo::builder()
                .max_sets(vulkan.swapchain.len() as u32)
                .pool_sizes(&pool_sizes);
            vulkan.device.create_descriptor_pool(&create_info, None)?
        };
        let descriptor_sets = unsafe {
            let set_layouts = [vulkan.descriptor_set_layout].repeat(vulkan.swapchain.len());
            let create_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            vulkan.device.allocate_descriptor_sets(&create_info)?
        };
        // Register descriptor pool with vulkan for later cleanup
        vulkan.descriptor_pools.borrow_mut().push(descriptor_pool);

        // Attach each uniform buffer to corresponding descriptor set
        let buffer_descriptors = uniform_buffers.iter().map(|buf| {
            [DescriptorBufferInfo::builder()
                .buffer(*buf)
                .offset(0)
                .range(4)
                .build()]
        }).collect::<Vec<_>>();
        let descriptor_writes = descriptor_sets.iter().enumerate().map(|(i, set)| {
            WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_descriptors[i])
                .build()
        }).collect::<Vec<_>>();
        unsafe { vulkan.device.update_descriptor_sets(&descriptor_writes, &[]) };

        Ok (System {
            planets,
            vertex_buffer, index_buffer, index_count: sphere.indices.len() as u32,
            stars: stars?, sun: sun?,
            uniform_allocations, descriptor_sets
        })
    }

    // TODO return result?
    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, view: Matrix4<f32>, view_rot: Matrix4<f32>, projection: Matrix4<f32>, time: f32) {
        unsafe {
            // Update brightness uniform
            let alloc = &self.uniform_allocations[vulkan.swapchain.index];
            let dst = vulkan.device.map_memory(alloc.memory, alloc.offset, alloc.size, MemoryMapFlags::empty()).expect("Failed to map memory");
            *(dst as *mut f32) = ((time / 8.0).sin() / 4.0) + 0.75;
            vulkan.device.unmap_memory(alloc.memory);

            // Bind descriptor set for uniform
            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                0, &[self.descriptor_sets[vulkan.swapchain.index]], &[]);

            // Bind model data
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmdbuf, self.index_buffer, 0, IndexType::UINT16);

            // Render skybox
            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.stars.descriptor_set], &[]);

            let data = 0u32.to_ne_bytes(); // Turn lighting off for skybox
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::FRAGMENT, 128, &data);

            let model = Scale3::from([200.0, 200.0, 200.0]).to_homogeneous();
            let vp = projection * view_rot; // Skybox ignores camera translation
            let data = [model.as_slice(), vp.as_slice()].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_draw_indexed(cmdbuf, self.index_count, 1, 0, 0, 0);

            // Render sun
            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.sun.descriptor_set], &[]);

            let model = Rotation3::from_axis_angle(&Vector3::y_axis(), time / 40.0).to_homogeneous();
            let vp = projection * view;
            let data = [model.as_slice(), vp.as_slice()].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_draw_indexed(cmdbuf, self.index_count, 1, 0, 0, 0);

            // Render planets
            let data = 1u32.to_ne_bytes(); // Turn lighting back on for planets
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::FRAGMENT, 128, &data);
            
            for planet in &self.planets {
                planet.render(vulkan, cmdbuf, vp.as_slice(), self.index_count, time);
            }
        };
    }
}
