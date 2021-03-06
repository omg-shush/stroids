use std::error::Error;
use std::slice;

use ash::vk::{ImageCreateInfo, Extent3D, ImageCreateFlags, Format, ImageType, ImageLayout, SampleCountFlags, ImageUsageFlags, MemoryMapFlags, CommandBufferBeginInfo, CommandBufferUsageFlags, BufferUsageFlags, DeviceSize, ImageViewCreateInfo, ImageViewType, ImageAspectFlags, ImageSubresourceRange, BufferImageCopy, ImageSubresourceLayers, PipelineStageFlags, DependencyFlags, ImageMemoryBarrier, AccessFlags, SubmitInfo, FenceCreateInfo, SamplerCreateInfo, Filter, DescriptorPoolSize, DescriptorType, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo, DescriptorImageInfo, WriteDescriptorSet, DescriptorSet, DescriptorSetLayoutBinding, ShaderStageFlags, DescriptorSetLayoutCreateInfo};
use image::{GenericImageView};
use image::io::Reader;
use nalgebra::{Vector3, vector};

use crate::vulkan::vulkan_instance::VulkanInstance;

pub struct Texture {
    pub descriptor_set: DescriptorSet
}

impl Texture {
    pub fn new(vulkan: &VulkanInstance, file: &str) -> Result<Texture, Box<dyn Error>> {
        // let file = if !cfg!(debug_assertions) {
        //    file.replace("2", "8") // Select high-res textures
        // } else { file.to_owned() };
        println!("Loading texture {:}", file);
        let image = Reader::open(file)?.decode()?;

        // Create vulkan image
        let create_info = ImageCreateInfo::builder()
            .array_layers(1)
            .extent(Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1
            })
            .flags(ImageCreateFlags::empty())
            .format(Format::R8G8B8A8_SRGB)
            .image_type(ImageType::TYPE_2D)
            .mip_levels(1)
            .samples(SampleCountFlags::TYPE_1)
            .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED)
            .build();
        let texture = vulkan.allocator.allocate_image(&vulkan.device, &create_info)?;

        // Create image view
        let subresource_range = ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
        let create_info = ImageViewCreateInfo::builder()
            .image(texture)
            .view_type(ImageViewType::TYPE_2D)
            .format(Format::R8G8B8A8_SRGB)
            .subresource_range(subresource_range);
        let view = unsafe { vulkan.device.create_image_view(&create_info, None)? };
        vulkan.allocator.image_views.borrow_mut().push(view);

        // Create sampler
        let create_info = SamplerCreateInfo::builder()
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR);
        let sampler = unsafe { vulkan.device.create_sampler(&create_info, None)? };
        vulkan.allocator.samplers.borrow_mut().push(sampler);

        // Construct descriptor set
        let descriptor_pool = unsafe {
            let pool_sizes = [DescriptorPoolSize {
                ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1
            }];
            let create_info = DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            vulkan.device.create_descriptor_pool(&create_info, None)?
        };
        let descriptor_set = unsafe {
            let set_layouts = [vulkan.image_descriptor_set_layout];
            let create_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            vulkan.device.allocate_descriptor_sets(&create_info)?[0]
        };
        // Register descriptor pool with vulkan for later cleanup
        vulkan.descriptor_pools.borrow_mut().push(descriptor_pool);

        // Add image/sampler to descriptor set
        let image_info = DescriptorImageInfo::builder()
            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)
            .build();
        let write_op = WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&[image_info])
            .build();
        unsafe { vulkan.device.update_descriptor_sets(&[write_op], &[]) };

        // Initialize staging buffer
        // TODO clean up!
        let (staging_buffer, staging_allocation) =
            vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::TRANSFER_SRC, (image.width() * image.height() * 4) as DeviceSize)?;
        unsafe {
            let ptr = vulkan.device.map_memory(staging_allocation.memory, staging_allocation.offset, staging_allocation.size, MemoryMapFlags::empty())?;
            let data = image.pixels().map(|p| (p.2).0).flatten().collect::<Vec<_>>();
            (ptr as *mut u8).copy_from_nonoverlapping(data.as_ptr(), data.len());
            vulkan.device.unmap_memory(staging_allocation.memory);
        }

        // Transfer from staging buffer & transition image layout
        let cmdbuf = vulkan.graphics_command_buffers[0];
        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy_op = BufferImageCopy::builder()
            .image_extent(Extent3D { width: image.width(), height: image.height(), depth: 1 })
            .image_subresource(ImageSubresourceLayers {
                aspect_mask: ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            });
        let barrier_transfer = ImageMemoryBarrier::builder()
            .image(texture)
            .src_access_mask(AccessFlags::empty())
            .dst_access_mask(AccessFlags::TRANSFER_WRITE)
            .old_layout(ImageLayout::UNDEFINED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .subresource_range(subresource_range);
        let barrier_texture = ImageMemoryBarrier::builder()
            .image(texture)
            .src_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ)
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .subresource_range(subresource_range);
        unsafe {
            vulkan.device.begin_command_buffer(cmdbuf, &begin_info)?;
            // Transition to transfer destination
            vulkan.device.cmd_pipeline_barrier(cmdbuf,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*barrier_transfer]);
            // Perform transfer
            vulkan.device.cmd_copy_buffer_to_image(cmdbuf, staging_buffer, texture, ImageLayout::TRANSFER_DST_OPTIMAL, &[*copy_op]);
            // Transition to shader texture
            vulkan.device.cmd_pipeline_barrier(cmdbuf,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::FRAGMENT_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*barrier_texture]);
            vulkan.device.end_command_buffer(cmdbuf)?;
            let submit_info = *SubmitInfo::builder().command_buffers(&[cmdbuf]);
            let fence = vulkan.device.create_fence(&FenceCreateInfo::default(), None)?;
            vulkan.device.queue_submit(vulkan.graphics_queue, &[submit_info], fence)?;
            vulkan.device.wait_for_fences(&[fence], true, std::u64::MAX)?;
            vulkan.device.destroy_fence(fence, None);
        };

        Ok (Texture { descriptor_set })
    }

    pub fn noise(vulkan: &VulkanInstance, dimensions: Vector3<u32>, noise: Box<dyn Fn(Vector3<f32>) -> f32>) -> Result<Texture, Box<dyn Error>> {
        // Create vulkan image
        let image_extent = Extent3D {
            width: dimensions[0],
            height: dimensions[1],
            depth: dimensions[2]
        };
        let create_info = ImageCreateInfo::builder()
            .array_layers(1)
            .extent(image_extent)
            .flags(ImageCreateFlags::empty())
            .format(Format::R32_SFLOAT)
            .image_type(ImageType::TYPE_3D)
            .mip_levels(1)
            .samples(SampleCountFlags::TYPE_1)
            .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::STORAGE)
            .build();
        let texture = vulkan.allocator.allocate_image(&vulkan.device, &create_info)?;

        // Create image view
        let subresource_range = ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
        let create_info = ImageViewCreateInfo::builder()
            .image(texture)
            .view_type(ImageViewType::TYPE_3D)
            .format(Format::R32_SFLOAT)
            .subresource_range(subresource_range);
        let view = unsafe { vulkan.device.create_image_view(&create_info, None)? };
        vulkan.allocator.image_views.borrow_mut().push(view);

        // Construct descriptor set
        let descriptor_pool = unsafe {
            let pool_sizes = [DescriptorPoolSize {
                ty: DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1
            }];
            let create_info = DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            vulkan.device.create_descriptor_pool(&create_info, None)?
        };
        let descriptor_set_layout = {
            let descriptor_bindings = [
                DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(ShaderStageFlags::COMPUTE)
                    .build()
            ];
            let create_info = DescriptorSetLayoutCreateInfo::builder()
                .bindings(&descriptor_bindings);
            unsafe { vulkan.device.create_descriptor_set_layout(&create_info, None)? }
        };
        let descriptor_set = unsafe {
            let set_layouts = [descriptor_set_layout];
            let create_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            vulkan.device.allocate_descriptor_sets(&create_info)?[0]
        };
        // Register descriptor pool with vulkan for later cleanup
        vulkan.descriptor_pools.borrow_mut().push(descriptor_pool);

        // Add image view to descriptor set
        let image_info = DescriptorImageInfo::builder()
            .image_layout(ImageLayout::GENERAL)
            .image_view(view)
            .build();
        let write_op = WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .image_info(&[image_info])
            .build();
        unsafe { vulkan.device.update_descriptor_sets(&[write_op], &[]) };

        // Cleanup image descriptor layout
        unsafe { vulkan.device.destroy_descriptor_set_layout(descriptor_set_layout, None) };

        // Initialize staging buffer
        // TODO clean up!
        let (staging_buffer, staging_allocation) =
            vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::TRANSFER_SRC, (dimensions.product() * 4) as DeviceSize)?;
        unsafe {
            let ptr = vulkan.device.map_memory(staging_allocation.memory, staging_allocation.offset, staging_allocation.size, MemoryMapFlags::empty())?;
            let data = slice::from_raw_parts_mut(ptr as *mut f32, dimensions.product() as usize);
            for x in 0..dimensions[0] {
                for y in 0..dimensions[1] {
                    for z in 0..dimensions[2] {
                        let index = x * dimensions[1] * dimensions[2] + y * dimensions[2] + z;
                        data[index as usize] = noise(vector![x, y, z].cast::<f32>());
                    }
                }
            }
            vulkan.device.unmap_memory(staging_allocation.memory);
        }

        // Transfer from staging buffer & transition image layout
        let cmdbuf = vulkan.graphics_command_buffers[0];
        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy_op = BufferImageCopy::builder()
            .image_extent(image_extent)
            .image_subresource(ImageSubresourceLayers {
                aspect_mask: ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            });
        let barrier_transfer = ImageMemoryBarrier::builder()
            .image(texture)
            .src_access_mask(AccessFlags::empty())
            .dst_access_mask(AccessFlags::TRANSFER_WRITE)
            .old_layout(ImageLayout::UNDEFINED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .subresource_range(subresource_range);
        let barrier_texture = ImageMemoryBarrier::builder()
            .image(texture)
            .src_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ)
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ImageLayout::GENERAL)
            .subresource_range(subresource_range);
        unsafe {
            vulkan.device.begin_command_buffer(cmdbuf, &begin_info)?;
            // Transition to transfer destination
            vulkan.device.cmd_pipeline_barrier(cmdbuf,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*barrier_transfer]);
            // Perform transfer
            vulkan.device.cmd_copy_buffer_to_image(cmdbuf, staging_buffer, texture, ImageLayout::TRANSFER_DST_OPTIMAL, &[*copy_op]);
            // Transition to shader texture
            vulkan.device.cmd_pipeline_barrier(cmdbuf,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[*barrier_texture]);
            vulkan.device.end_command_buffer(cmdbuf)?;
            let submit_info = *SubmitInfo::builder().command_buffers(&[cmdbuf]);
            let fence = vulkan.device.create_fence(&FenceCreateInfo::default(), None)?;
            vulkan.device.queue_submit(vulkan.graphics_queue, &[submit_info], fence)?;
            vulkan.device.wait_for_fences(&[fence], true, std::u64::MAX)?;
            vulkan.device.destroy_fence(fence, None);
        };

        Ok (Texture { descriptor_set })
    }
}
