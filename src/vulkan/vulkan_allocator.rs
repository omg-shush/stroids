use std::cell::RefCell;
use std::error::Error;

use ash::{Device, Instance};
use ash::vk::{PhysicalDevice, Buffer, DeviceMemory, MemoryPropertyFlags, PhysicalDeviceMemoryProperties, MemoryAllocateInfo, BufferCreateInfo, SharingMode, BufferUsageFlags, DeviceSize, BindBufferMemoryInfo, MemoryRequirements, Image, ImageCreateInfo, ImageView, Sampler};

pub struct VulkanAllocator {
    memory_properties: PhysicalDeviceMemoryProperties,
    buffers: RefCell<Vec<Buffer>>,
    images: RefCell<Vec<Image>>,
    pub image_views: RefCell<Vec<ImageView>>, // TODO this and samplers should be somewhere better
    pub samplers: RefCell<Vec<Sampler>>,
    device_allocations: RefCell<Vec<DeviceAllocation>>
}

struct DeviceAllocation {
    pub memory: DeviceMemory,
    pub index: u32,
    pub size: DeviceSize,
    pub free_offset: DeviceSize
}

pub struct Allocation {
    pub memory: DeviceMemory,
    pub offset: DeviceSize,
    pub size: DeviceSize
}

impl VulkanAllocator {
    pub fn new(instance: &Instance, card: &PhysicalDevice) -> Result<VulkanAllocator, Box<dyn Error>> {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(*card) };
        Ok (VulkanAllocator {
            memory_properties, buffers: vec![].into(), images: vec![].into(), image_views: vec![].into(), samplers: vec![].into(), device_allocations: vec![].into()
        })
    }

    pub fn drop(&self, device: &Device) {
        unsafe {
            self.buffers.borrow().iter().for_each(|buffer| device.destroy_buffer(*buffer, None));
            self.samplers.borrow().iter().for_each(|sampler| device.destroy_sampler(*sampler, None));
            self.image_views.borrow().iter().for_each(|image_view| device.destroy_image_view(*image_view, None));
            self.images.borrow().iter().for_each(|image| device.destroy_image(*image, None));
            self.device_allocations.borrow().iter().for_each(|alloc| device.free_memory(alloc.memory, None));
        }
    }

    // Returns the given ptr rounded up to the next multiple of the given alignment
    fn align_up(ptr: DeviceSize, alignment: DeviceSize) -> DeviceSize {
        if ptr == 0 {
            0
        } else {
            (((ptr - 1) / alignment) + 1) * alignment
        }
    }

    pub fn allocate_memory(&self, device: &Device, reqs: MemoryRequirements, props: MemoryPropertyFlags) -> Result<Allocation, Box<dyn Error>> {
        let size = reqs.size;
        let alignment = reqs.alignment;
        let mut device_allocations = self.device_allocations.borrow_mut();

        // Search for a device allocation with enough free space
        // Otherwise, create a new allocation
        let next_allocation_index = device_allocations.len();
        let device_allocation_index = device_allocations.iter_mut()
            .position(|da| {
                ((1 << da.index) & reqs.memory_type_bits > 0)
                    && (self.memory_properties.memory_types[da.index as usize].property_flags.contains(props))
                    && (da.size - VulkanAllocator::align_up(da.free_offset, alignment) >= size)
            }).ok_or("No suitable existing allocation found".into()).or_else::<Box<dyn Error>, _>(|_: Box<dyn Error>| {
                let device_size = size.max(1 << 27);
                let memory_type_index = self.get_memory_type_index(props, reqs.memory_type_bits)?;
                let create_info = MemoryAllocateInfo::builder()
                    .allocation_size(device_size)
                    .memory_type_index(memory_type_index);
                let memory = unsafe { device.allocate_memory(&create_info, None)? };
                let new_alloc = DeviceAllocation { memory, index: memory_type_index, size: device_size, free_offset: 0 };
                device_allocations.push(new_alloc);
                Ok (next_allocation_index)
            })?;
        let device_allocation = &mut device_allocations[device_allocation_index];

        let alloc = Allocation { memory: device_allocation.memory, offset: VulkanAllocator::align_up(device_allocation.free_offset, alignment), size };
        // Remove used space from allocation
        device_allocation.free_offset = VulkanAllocator::align_up(device_allocation.free_offset, alignment) + size;
        Ok (alloc)
    }

    // Constructs a single buffer, allocates device memory, and binds the two together
    // Tracks the buffer for later cleanup. TODO create some sort of "memory pool" so certain memory can be freed mid-execution
    pub fn allocate_buffer(&self, device: &Device, usage: BufferUsageFlags, size: DeviceSize) -> Result<(Buffer, Allocation), Box<dyn Error>> {
        unsafe {
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size as u64)
                .usage(usage);
            let buffer = device.create_buffer(&create_info, None)?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            let allocation = self.allocate_memory(device, reqs, MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)?;
            device.bind_buffer_memory(buffer, allocation.memory, allocation.offset)?;
            self.buffers.borrow_mut().push(buffer);
            Ok ((buffer, allocation))
        }
    }

    // Allocates a ring of identical buffers of the given count, each with the given size
    pub fn allocate_buffer_chain(&self, device: &Device, usage: BufferUsageFlags, size: DeviceSize, count: usize) -> Result<(Vec<Buffer>, Vec<Allocation>), Box<dyn Error>> {
        unsafe {
            let create_info = BufferCreateInfo::builder()
                .sharing_mode(SharingMode::EXCLUSIVE)
                .size(size)
                .usage(usage);
            let mut buffers = Vec::new();
            for _ in 0..count {
                let buf = device.create_buffer(&create_info, None)?;
                buffers.push(buf);
            }
            let allocations = {
                buffers.iter()
                    .map(|b| device.get_buffer_memory_requirements(*b))
                    .map(|r| self.allocate_memory(device, r, MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT))
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
                device.bind_buffer_memory2(&bind_infos)?;
            };
            self.buffers.borrow_mut().append(&mut buffers.to_vec());
            Ok ((buffers, allocations))
        }
    }

    pub fn allocate_image(&self, device: &Device, create_info: &ImageCreateInfo) -> Result<Image, Box<dyn Error>> {
        let image = unsafe { device.create_image(create_info, None)? };
        self.images.borrow_mut().push(image);
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let allocation = self.allocate_memory(device, reqs, MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|_| self.allocate_memory(device, reqs, MemoryPropertyFlags::empty()))?;
        unsafe { device.bind_image_memory(image, allocation.memory, allocation.offset)? };
        Ok (image)
    }

    fn get_memory_type_index(&self, required_properties: MemoryPropertyFlags, valid_types: u32) -> Result<u32, Box<dyn Error>> {
        for i in 0..self.memory_properties.memory_type_count {
            let is_valid = ((1 << i) & valid_types) > 0;
            let memory_type = self.memory_properties.memory_types[i as usize];
            if is_valid && memory_type.property_flags.contains(required_properties) {
                return Ok (i);
            }
        }
        Err ("Could not find memory type with required properties".into())
    }
}
