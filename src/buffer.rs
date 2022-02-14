use std::error::Error;
use std::mem::size_of;

use ash::vk::{Buffer, BufferUsageFlags, MemoryMapFlags};

use crate::vulkan::vulkan_allocator::Allocation;
use crate::vulkan::vulkan_instance::VulkanInstance;

pub struct DynamicBuffer {
    pub buffer: Buffer,
    allocation: Allocation,
    pub len: u32
}

impl DynamicBuffer {
    pub fn new<T>(vulkan: &VulkanInstance, data: &[T]) -> Result<DynamicBuffer, Box<dyn Error>> {
        let (buffer, allocation) = vulkan.allocator.allocate_buffer(
            &vulkan.device, BufferUsageFlags::VERTEX_BUFFER, (data.len() * size_of::<T>()) as u64)?;
        unsafe {
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut T).copy_from_nonoverlapping(data.as_ptr(), data.len());
            vulkan.device.unmap_memory(allocation.memory);
        }
        Ok (DynamicBuffer { buffer, allocation, len: data.len() as u32 })
    }
}
