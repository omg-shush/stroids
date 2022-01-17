use std::time::Instant;
use std::slice;

use ash::vk::{DeviceMemory, CommandBuffer, Buffer, BufferCreateInfo, SharingMode, BufferUsageFlags, MemoryAllocateInfo, MemoryPropertyFlags, MemoryMapFlags, ShaderStageFlags, DeviceSize, BindBufferMemoryInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, DescriptorSetAllocateInfo, DescriptorSet, WriteDescriptorSet, DescriptorBufferInfo, PipelineBindPoint};

use crate::vulkan::vulkan_instance::VulkanInstance;

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
    device_memory: DeviceMemory,
    uniform_buffers: Vec<Buffer>,
    uniform_memory: DeviceMemory,
    uniform_offsets: Vec<u64>,
    descriptor_sets: Vec<DescriptorSet>
}

impl System {
    pub fn new(vulkan: &VulkanInstance) -> System {
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

        let memory_type_index = vulkan.get_memory_type_index( // TODO try to get device local too
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT).expect("TODO");

        let data = planets.iter()
            .map(|p| [p.color[0], p.color[1], p.color[2], p.radius, p.orbit, p.year])
            .flatten()
            .collect::<Vec<_>>();
        let size = (data.len() * 4) as DeviceSize;

        let (vertex_buffer, device_memory) = unsafe {
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size)
                .usage(BufferUsageFlags::VERTEX_BUFFER);
            let vertex_buffer = vulkan.device.create_buffer(&create_info, None).expect("Failed to create buffer");
            let size = vulkan.device.get_buffer_memory_requirements(vertex_buffer).size;
            let device_memory = {
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(size)
                    .memory_type_index(memory_type_index);
                vulkan.device.allocate_memory(&create_info, None).expect("Failed to allocate device memory")
            };
            vulkan.device.bind_buffer_memory(vertex_buffer, device_memory, 0).expect("Failed to bind buffer memory");

            // Write to buffer
            {
                let dst = vulkan.device.map_memory(device_memory, 0, size, MemoryMapFlags::empty()).expect("Failed to map memory");
                (dst as *mut f32).copy_from_nonoverlapping(data.as_ptr(), data.len());
                vulkan.device.unmap_memory(device_memory);
            }

            (vertex_buffer, device_memory)
        };
        let (uniform_buffers, uniform_memory, uniform_offsets) = unsafe {
            let size = 4; // uniform contains 1 float
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size)
                .usage(BufferUsageFlags::UNIFORM_BUFFER);
            let mut uniform_buffers = Vec::new();
            for _ in 0..vulkan.swapchain.len() {
                let buf = vulkan.device.create_buffer(&create_info, None).expect("Failed to create buffer");
                uniform_buffers.push(buf);
            }
            let mut uniform_offsets = Vec::new();
            let uniform_memory = {
                let reqs = uniform_buffers.iter().map(|b| vulkan.device.get_buffer_memory_requirements(*b));
                let mut ptr = 0;
                for r in reqs {
                    ptr = if ptr == 0 { 0 } else { (((ptr - 1) / r.alignment) + 1) * r.alignment };
                    uniform_offsets.push(ptr);
                    ptr += r.size;
                }
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(ptr)
                    .memory_type_index(memory_type_index);
                vulkan.device.allocate_memory(&create_info, None).expect("Failed to allocate memory")
            };
            { // bind each uniform buffer to a region in the device memory
                let mut bind_infos = Vec::new();
                for i in 0..uniform_buffers.len() {
                    bind_infos.push(BindBufferMemoryInfo::builder()
                        .buffer(uniform_buffers[i])
                        .memory(uniform_memory)
                        .memory_offset(uniform_offsets[i])
                        .build());
                }
                vulkan.device.bind_buffer_memory2(&bind_infos).expect("Failed to bind uniform buffers");
            };
            (uniform_buffers, uniform_memory, uniform_offsets)
        };
        // Register buffer/memory with vulkan for later cleanup
        uniform_buffers.iter().for_each(|b| vulkan.buffers.borrow_mut().push(*b));
        vulkan.buffers.borrow_mut().push(vertex_buffer);
        vulkan.memory.borrow_mut().push(device_memory);
        vulkan.memory.borrow_mut().push(uniform_memory);

        // Construct descriptor sets to bind uniform buffers
        let descriptor_pool = unsafe {
            let pool_sizes = [DescriptorPoolSize {
                ty: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: vulkan.swapchain.len() as u32
            }];
            let create_info = DescriptorPoolCreateInfo::builder()
                .max_sets(vulkan.swapchain.len() as u32)
                .pool_sizes(&pool_sizes);
            vulkan.device.create_descriptor_pool(&create_info, None).expect("Failed to create descriptor pool")
        };
        let descriptor_sets = unsafe {
            let set_layouts = [vulkan.descriptor_set_layout].repeat(vulkan.swapchain.len());
            let create_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            vulkan.device.allocate_descriptor_sets(&create_info).expect("Failed to allocate descriptor sets")
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

        System { start: Instant::now(), planets, vertex_buffer, device_memory,
            uniform_buffers, uniform_memory, uniform_offsets, descriptor_sets }
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer) {
        unsafe {
            let time = (Instant::now() - self.start).as_secs_f32();

            // Update brightness uniform
            let dst = vulkan.device.map_memory(self.uniform_memory, self.uniform_offsets[vulkan.swapchain.index], 4, MemoryMapFlags::empty()).expect("Failed to map memory");
            *(dst as *mut f32) = ((time / 8.0).sin() + 1.0) / 2.0;
            vulkan.device.unmap_memory(self.uniform_memory);

            // Bind that uniform's descriptor set
            vulkan.device.cmd_bind_descriptor_sets(
                cmdbuf,
                PipelineBindPoint::GRAPHICS,
                vulkan.pipeline_layout,
                0,
                &[self.descriptor_sets[vulkan.swapchain.index]],
                &[]);

            let data = slice::from_raw_parts([time].as_ptr() as *const u8, 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, data);
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            vulkan.device.cmd_draw(cmdbuf, self.planets.len() as u32, 1, 0, 0);
        };
    }
}
