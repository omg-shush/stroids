use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use std::slice;
use std::error::Error;
use std::mem::size_of;

use ash::vk::{CommandBuffer, Buffer, BufferUsageFlags, MemoryMapFlags, ShaderStageFlags, DeviceSize, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, DescriptorSetAllocateInfo, DescriptorSet, WriteDescriptorSet, DescriptorBufferInfo, PipelineBindPoint, IndexType};
use nalgebra::{Rotation3, Scale3, Perspective3, Translation3, Vector3};
use obj::{load_obj, Obj, TexturedVertex};

use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;
use crate::vulkan::vulkan_allocator::Allocation;

pub struct Planet {
    color: [f32; 3],
    radius: f32,
    orbit: f32,
    year: f32
}

pub struct System {
    start: Instant,
    planets: Vec<Planet>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
    texture: Texture,
    uniform_allocations: Vec<Allocation>,
    descriptor_sets: Vec<DescriptorSet>
}

impl System {
    pub fn new(vulkan: &VulkanInstance) -> Result<System, Box<dyn Error>> {
        let planets = vec![
            Planet {
                color: [1.0, 1.0, 0.0],
                radius: 40.0,
                orbit: 0.01,
                year: 20.0
            },
            Planet {
                color: [0.6, 0.1, 0.1],
                radius: 3.0,
                orbit: 0.15,
                year: 0.4
            },
            Planet {
                color: [1.0, 0.5, 0.5],
                radius: 6.0,
                orbit: 0.3,
                year: 1.0
            },
            Planet {
                color: [0.5, 1.0, 0.5],
                radius: 10.0,
                orbit: 0.5,
                year: 2.5
            },
            Planet {
                color: [0.3, 0.3, 0.8],
                radius: 8.0,
                orbit: 0.65,
                year: 3.5
            },
            Planet {
                color: [0.5, 0.5, 1.0],
                radius: 4.0,
                orbit: 0.8,
                year: 5.0
            }
        ];

        let file = BufReader::new(File::open("res/sphere.obj")?);
        let wall: Obj<TexturedVertex> = load_obj(file)?; // TODO TexturedVertex
        // let data = planets.iter()
        //     .map(|p| [
        //         [0.0, 0.0, 0.0, p.color[0], p.color[1], p.color[2]],
        //         [1.0, 0.0, 0.0, p.color[0], p.color[1], p.color[2]],
        //         [1.0, 1.0, 0.0, p.color[0], p.color[1], p.color[2]],
        //         [1.0, 1.0, 0.0, p.color[0], p.color[1], p.color[2]],
        //         [0.0, 1.0, 0.0, p.color[0], p.color[1], p.color[2]],
        //         [0.0, 0.0, 0.0, p.color[0], p.color[1], p.color[2]]
        //     ]).flatten()
        //     .flatten()
        //     .collect::<Vec<_>>();
        // let size = (data.len() * 4) as DeviceSize;

        // Load texture
        let texture = Texture::new(&vulkan, "res/1k_venus_surface.png")?;

        // Allocate & write vertex buffer
        let vertices = wall.vertices.iter().map(|v| {
            let mut vec = [v.position, v.normal].concat();
            vec.extend_from_slice(&v.texture[..2]);
            vec
        }).flatten().collect::<Vec<_>>();
        let (vertex_buffer, _vertex_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(BufferUsageFlags::VERTEX_BUFFER, (vertices.len() * 4) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut f32).copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        // Allocate & write index buffer
        let (index_buffer, _index_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(BufferUsageFlags::INDEX_BUFFER, (wall.indices.len() * 2) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut u16).copy_from_nonoverlapping(wall.indices.as_ptr(), wall.indices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        // Allocate uniform buffers
        let (uniform_buffers, uniform_allocations) = vulkan.allocator.allocate_buffer_chain(
            BufferUsageFlags::UNIFORM_BUFFER, size_of::<f32>() as DeviceSize, vulkan.swapchain.len())?;

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
            start: Instant::now(), planets,
            vertex_buffer, index_buffer, index_count: wall.indices.len() as u32,
            texture, uniform_allocations, descriptor_sets
        })
    }

    // TODO return result?
    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer) {
        unsafe {
            let time = (Instant::now() - self.start).as_secs_f32();

            // Update brightness uniform
            let alloc = &self.uniform_allocations[vulkan.swapchain.index];
            let dst = vulkan.device.map_memory(alloc.memory, alloc.offset, alloc.size, MemoryMapFlags::empty()).expect("Failed to map memory");
            *(dst as *mut f32) = ((time / 8.0).sin() / 4.0) + 0.75;
            vulkan.device.unmap_memory(alloc.memory);

            // Bind descriptor sets for uniform & image/samplers
            vulkan.device.cmd_bind_descriptor_sets(
                cmdbuf,
                PipelineBindPoint::GRAPHICS,
                vulkan.pipeline_layout,
                0,
                &[self.descriptor_sets[vulkan.swapchain.index], self.texture.descriptor_set],
                &[]);

            // Push constants
            let model = {
                Rotation3::from_axis_angle(&Vector3::y_axis(), time / 12.0).to_homogeneous()
            };
            let view = {
                let pos = Vector3::from([0.0, (time / 4.0).sin() * 0.3, -2.5]);
                let t = Translation3::from(-1.0 * pos);
                let dir = -1.0 * pos;
                let r = Rotation3::look_at_rh(&dir, &Vector3::from([0.0, 1.0, 0.0]));
                let s = Scale3::identity();
                r.to_homogeneous() * s.to_homogeneous() * t.to_homogeneous()
            };
            let projection = {
                Perspective3::new(1080.0 / 720.2, 70f32.to_radians(), 0.1, 100.0).to_homogeneous()
            };
            let vp = projection * view * model;
            let data = slice::from_raw_parts(vp.as_ptr() as *const u8, 4 * 4 * 4); // 4x4 f32 matrix
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, data);

            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmdbuf, self.index_buffer, 0, IndexType::UINT16);
            vulkan.device.cmd_draw_indexed(cmdbuf, self.index_count, 1, 0, 0, 0);
        };
    }
}
