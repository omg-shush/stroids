use ash::Device;
use ash::vk::{DeviceMemory, CommandBuffer, Buffer, BufferCreateInfo, BufferCreateFlags, SharingMode, BufferUsageFlags, MemoryAllocateInfo, MemoryPropertyFlags, MemoryMapFlags};

use crate::vulkan_instance::VulkanInstance;

pub struct Planet {
    color: [f32; 3],
    radius: f32,
    orbit: f32,
    year: f32
}

pub struct System {
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
        System { planets, vertex_buffer, device_memory }
    }

    pub fn render(&self, device: &Device, cmdbuf: CommandBuffer) {
        unsafe {
            device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            device.cmd_draw(cmdbuf, 4, 1, 0, 0);
        };
    }
}
