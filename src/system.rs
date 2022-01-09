use std::time::Instant;
use std::slice;

use ash::vk::{DeviceMemory, CommandBuffer, Buffer, BufferCreateInfo, BufferCreateFlags, SharingMode, BufferUsageFlags, MemoryAllocateInfo, MemoryPropertyFlags, MemoryMapFlags, PipelineBindPoint, DescriptorSet, ShaderStageFlags};

use crate::vulkan_instance::VulkanInstance;

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
    device_memory: DeviceMemory
}

impl System {
    pub fn new(vulkan: &mut VulkanInstance) -> System {
        let planets = vec![
            Planet {
                color: [1.0, 0.0, 0.0],
                radius: 5.0,
                orbit: 0.1,
                year: 2.0
            },
            Planet {
                color: [0.0, 1.0, 0.0],
                radius: 3.0,
                orbit: 0.5,
                year: 5.0
            },
            Planet {
                color: [0.0, 0.0, 1.0],
                radius: 1.0,
                orbit: 0.8,
                year: 10.0
            }
        ];

        // TODO calculate using ***PHYSICS***
        let positions = [ 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5 ];

        let (vertex_buffer, device_memory) = unsafe {
            let create_info = BufferCreateInfo::builder()
                .queue_family_indices(&vulkan.queue_family_indices) // ignored in SharingMode::EXCLUSIVE
                .flags(BufferCreateFlags::empty())
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(4096) // TODO
                .usage(BufferUsageFlags::VERTEX_BUFFER);
            let vertex_buffer = vulkan.device.create_buffer(&create_info, None).expect("Failed to create buffer");
            let device_memory = {
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(4096) // TODO
                    .memory_type_index(vulkan.get_memory_type_index( // TODO try to get device local too
                        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT).expect("TODO"));
                vulkan.device.allocate_memory(&create_info, None).expect("Failed to allocate device memory")
            };
            vulkan.device.bind_buffer_memory(vertex_buffer, device_memory, 0).expect("Failed to bind buffer memory"); // TODO offset???

            // Write to buffer
            {
                let dst = vulkan.device.map_memory(device_memory, 0, 4096, MemoryMapFlags::empty()).expect("Failed to map memory");
                (dst as *mut f32).copy_from_nonoverlapping(positions.as_ptr(), positions.len());
                vulkan.device.unmap_memory(device_memory);
            }

            (vertex_buffer, device_memory)
        };
        // Register buffer/memory with vulkan for later cleanup
        vulkan.buffers.push(vertex_buffer);
        vulkan.memory.push(device_memory);
        System { start: Instant::now(), planets, vertex_buffer, device_memory }
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer) {
        unsafe {
            // let descriptor_set = vulkan.device.create_descr
            // vulkan.device.cmd_bind_descriptor_sets(cmdbuf,
            //     PipelineBindPoint::GRAPHICS,
            //     vulkan.pipeline_layout,
            //     0,
            //     descriptor_sets,
            //     0);
            let time = (Instant::now() - self.start).as_secs_f32();
            let data = slice::from_raw_parts([time].as_ptr() as *const u8, 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, data);
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            vulkan.device.cmd_draw(cmdbuf, 4, 1, 0, 0);
        };
    }
}
