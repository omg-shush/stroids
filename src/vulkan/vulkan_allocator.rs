use std::cell::RefCell;
use std::error::Error;

use ash::{Device, Instance};
use ash::vk::{PhysicalDevice, Buffer, DeviceMemory, MemoryPropertyFlags, PhysicalDeviceMemoryProperties, MemoryAllocateInfo, BufferCreateInfo, SharingMode, BufferUsageFlags, DeviceSize, BindBufferMemoryInfo};

pub struct VulkanAllocator {
    device: Device,
    memory_properties: PhysicalDeviceMemoryProperties,
    buffers: RefCell<Vec<Buffer>>,
    device_allocations: RefCell<Vec<DeviceAllocation>>
}

struct DeviceAllocation {
    pub memory: DeviceMemory,
    pub size: DeviceSize,
    pub free_offset: DeviceSize
}

pub struct Allocation {
    pub memory: DeviceMemory,
    pub offset: DeviceSize,
    pub size: DeviceSize
}

impl Drop for VulkanAllocator {
    fn drop(&mut self) {
        unsafe {
            self.buffers.borrow().iter().for_each(|buffer| self.device.destroy_buffer(*buffer, None));
            self.device_allocations.borrow().iter().for_each(|alloc| self.device.free_memory(alloc.memory, None));
        }
    }
}

impl VulkanAllocator {
    pub fn new(instance: &Instance, card: &PhysicalDevice, device: &Device) -> Result<VulkanAllocator, Box<dyn Error>> {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(*card) };
        Ok (VulkanAllocator {
            device: device.clone(), memory_properties, buffers: vec![].into(), device_allocations: vec![].into()
        })
    }

    // Returns the given ptr rounded up to the next multiple of the given alignment
    fn align_up(ptr: DeviceSize, alignment: DeviceSize) -> DeviceSize {
        if ptr == 0 {
            0
        } else {
            (((ptr - 1) / alignment) + 1) * alignment
        }
    }

    fn allocate_memory(&self, size: DeviceSize, alignment: DeviceSize) -> Result<Allocation, Box<dyn Error>> {
        let mut device_allocations = self.device_allocations.borrow_mut();

        // Search for a device allocation with enough free space
        // Otherwise, create a new allocation
        let next_allocation_index = device_allocations.len();
        let device_allocation_index = device_allocations.iter_mut()
            .position(|da| {
                da.size - VulkanAllocator::align_up(da.free_offset, alignment) >= size
            }).unwrap_or_else(|| {
                let device_size = size.max(1 << 27);
                let memory_type_index = self.get_memory_type_index(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT).expect("Failed to find appropriate device memory");
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(device_size)
                    .memory_type_index(memory_type_index);
                let memory = unsafe { self.device.allocate_memory(&create_info, None).expect("Failed to allocate device memory") };
                let new_alloc = DeviceAllocation { memory, size: device_size, free_offset: 0 };
                device_allocations.push(new_alloc);
                next_allocation_index
            });
        let device_allocation = &mut device_allocations[device_allocation_index];

        let alloc = Allocation { memory: device_allocation.memory, offset: VulkanAllocator::align_up(device_allocation.free_offset, alignment), size };
        // Remove used space from allocation
        device_allocation.free_offset = VulkanAllocator::align_up(device_allocation.free_offset, alignment) + size;
        Ok (alloc)
    }

    // Constructs a single buffer, allocates device memory, and binds the two together
    // Tracks the buffer for later cleanup. TODO create some sort of "memory pool" so certain memory can be freed mid-execution
    pub fn allocate_buffer(&self, usage: BufferUsageFlags, size: DeviceSize) -> Result<(Buffer, Allocation), Box<dyn Error>> {
        unsafe {
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size as u64)
                .usage(usage);
            let buffer = self.device.create_buffer(&create_info, None)?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let allocation = self.allocate_memory(reqs.size, reqs.alignment)?;
            self.device.bind_buffer_memory(buffer, allocation.memory, allocation.offset)?;
            self.buffers.borrow_mut().push(buffer);
            Ok ((buffer, allocation))
        }
    }

    // Allocates a ring of identical buffers of the given count, each with the given size
    pub fn allocate_buffer_chain(&self, usage: BufferUsageFlags, size: DeviceSize, count: usize) -> Result<(Vec<Buffer>, Vec<Allocation>), Box<dyn Error>> {
        unsafe {
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size)
                .usage(usage);
            let mut buffers = Vec::new();
            for _ in 0..count {
                let buf = self.device.create_buffer(&create_info, None)?;
                buffers.push(buf);
            }
            let allocations = {
                buffers.iter()
                    .map(|b| self.device.get_buffer_memory_requirements(*b))
                    .map(|r| self.allocate_memory(r.size, r.alignment))
                    .collect::<Result<Vec<_>, _>>()?
            };
            { // bind each buffer to a region in the device memory
                let mut bind_infos = Vec::new();
                for i in 0..buffers.len() {
                    bind_infos.push(BindBufferMemoryInfo::builder()
                        .buffer(buffers[i])
                        .memory(allocations[i].memory)
                        .memory_offset(allocations[i].offset)
                        .build());
                }
                self.device.bind_buffer_memory2(&bind_infos)?;
            };
            self.buffers.borrow_mut().append(&mut buffers.to_vec());
            Ok ((buffers, allocations))
        }
    }

    fn get_memory_type_index(&self, required_properties: MemoryPropertyFlags) -> Result<u32, Box<dyn Error>> {
        for i in 0..self.memory_properties.memory_type_count {
            let memory_type = self.memory_properties.memory_types[i as usize];
            if memory_type.property_flags.contains(required_properties) {
                return Ok (i);
            }
        }
        Err ("Could not find memory type with required properties".into())
    }
}
