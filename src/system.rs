use std::time::Instant;
use std::slice;

use ash::vk::{DeviceMemory, CommandBuffer, Buffer, BufferCreateInfo, BufferCreateFlags, SharingMode, BufferUsageFlags, MemoryAllocateInfo, MemoryPropertyFlags, MemoryMapFlags, PipelineBindPoint, DescriptorSet, ShaderStageFlags, DeviceSize};

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

        let data = planets.iter()
            .map(|p| [p.color[0], p.color[1], p.color[2], p.radius, p.orbit, p.year])
            .flatten()
            .collect::<Vec<_>>();
        let size = (data.len() * 4) as DeviceSize;

        let (vertex_buffer, device_memory) = unsafe {
            let create_info = BufferCreateInfo::builder()
                .queue_family_indices(&vulkan.queue_family_indices) // ignored in SharingMode::EXCLUSIVE
                .flags(BufferCreateFlags::empty())
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size)
                .usage(BufferUsageFlags::VERTEX_BUFFER);
            let vertex_buffer = vulkan.device.create_buffer(&create_info, None).expect("Failed to create buffer");
            let size = size.max(vulkan.device.get_buffer_memory_requirements(vertex_buffer).size); // Meet minimum buffer size
            let device_memory = {
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(size)
                    .memory_type_index(vulkan.get_memory_type_index( // TODO try to get device local too
                        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT).expect("TODO"));
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
            vulkan.device.cmd_draw(cmdbuf, self.planets.len() as u32, 1, 0, 0);
        };
    }
}
