use std::cell::RefCell;
use std::error::Error;
use std::ffi::{CString, c_void, NulError};
use std::mem::ManuallyDrop;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain, Win32Surface, XlibSurface};
use ash::{vk, Entry, Instance, Device};
use ash::vk::{Format, Pipeline, RenderPass, Queue, PhysicalDevice, PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, QueueFlags, SurfaceFormatKHR, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, ImageLayout, SampleCountFlags, AttachmentReference, SubpassDescription, PipelineBindPoint, SubpassDependency, SUBPASS_EXTERNAL, PipelineStageFlags, AccessFlags, RenderPassCreateInfo, Extent2D, ShaderStageFlags, PipelineVertexInputStateCreateInfo, Rect2D, Offset2D, PipelineLayoutCreateInfo, PipelineLayout, CommandPool, CommandPoolCreateInfo, CommandPoolCreateFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferUsageFlags, ClearValue, ClearColorValue, RenderPassBeginInfo, SubpassContents, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, PushConstantRange, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DescriptorSetLayoutCreateInfo, DescriptorPool, ImageCreateInfo, Extent3D, ImageUsageFlags, ImageSubresourceRange, ImageAspectFlags, ImageViewType, ImageViewCreateInfo, ImageType, ClearDepthStencilValue};
use winit::window::Window;

use super::vulkan_pipeline::VulkanPipeline;
use super::vulkan_swapchain::VulkanSwapchain;
use super::vulkan_allocator::VulkanAllocator;
use super::vulkan_surface::VulkanSurface;

pub struct VulkanInstance {
    entry: Entry,
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    surface: ManuallyDrop<VulkanSurface>,
    pub device: Device,
    pub graphics_queue: Queue,
    transfer_queue: Queue,
    pub queue_indices: Vec<u32>,
    extent: Extent2D,
    pub swapchain: VulkanSwapchain,
    render_pass: RenderPass,
    pub descriptor_set_layout: DescriptorSetLayout,
    pub image_descriptor_set_layout: DescriptorSetLayout,
    graphics_pipeline: Pipeline,
    pub pipeline_layout: PipelineLayout,
    graphics_pool: CommandPool,
    transfer_pool: CommandPool,
    pub graphics_command_buffers: Vec<CommandBuffer>,
    pub allocator: VulkanAllocator,
    pub descriptor_pools: RefCell<Vec<DescriptorPool>>
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait for device idle");

            self.device.destroy_command_pool(self.graphics_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.allocator.drop(&self.device);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_pools.borrow().iter().for_each(|pool| self.device.destroy_descriptor_pool(*pool, None));
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_set_layout(self.image_descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain.drop(&self.device);
            self.device.destroy_device(None);
            ManuallyDrop::drop(&mut self.surface);
            self.debug_utils.destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }
}

unsafe extern "system" fn debug_utils_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug] [{}] [{}] {:?}", severity, ty, message);
    vk::FALSE
}

impl VulkanInstance {
    pub fn new(window: &Window) -> Result<VulkanInstance, Box<dyn Error>> {
        let entry = unsafe { Entry::load() }?;

        // Describe application
        let app_name = CString::new("'Stroids")?;
        let engine_name = CString::new("Astral Engine")?;
        let app_info = ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 2, 0))
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .application_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name);

        // Choose extensions & layers
        let supported_extensions: Vec<CString> = entry.enumerate_instance_extension_properties()?.into_iter().map(|ep| {
            CString::new(ep.extension_name.map(|i| i as u8).into_iter().take_while(|c| *c != 0).collect::<Vec<_>>())
        }).collect::<Result<Vec<CString>, _>>()?;
        let desired_extensions = vec![
            Surface::name().to_owned(),
            Win32Surface::name().to_owned(),
            XlibSurface::name().to_owned(),
            DebugUtils::name().to_owned()
        ];
        let extensions = desired_extensions.iter().filter(|e| supported_extensions.contains(e)).map(|e| e.as_ptr()).collect::<Vec<_>>();
        let layer_names = if cfg!(debug_assertions) {
            vec!["VK_LAYER_KHRONOS_validation"].into_iter().map(|s| CString::new(s)).collect::<Result<Vec<_>, NulError>>()?
        } else {
            vec![]
        };
        let layers = layer_names.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        
        // Construct instance
        let mut debug_create_info = DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(DebugUtilsMessageSeverityFlagsEXT::WARNING | DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(DebugUtilsMessageTypeFlagsEXT::GENERAL | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .pfn_user_callback(Some (debug_utils_callback));
        let create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers)
            .push_next(&mut debug_create_info);
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // Init debug callback
        let debug_utils = DebugUtils::new(&entry, &instance);
        let debug_utils_messenger = unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None)? };

        // Init platform-specific surface
        let surface = VulkanSurface::new(&entry, &instance, &window)?;

        // Init device & queues
        let (card, device, queue_family_indices) = VulkanInstance::select_device(&instance, &surface, &layers)?;
        let msaa_count = {
            let limits = unsafe { instance.get_physical_device_properties(card).limits };
            let bits = limits.framebuffer_color_sample_counts & limits.framebuffer_depth_sample_counts;
            [
                SampleCountFlags::TYPE_64,
                SampleCountFlags::TYPE_32,
                SampleCountFlags::TYPE_16,
                SampleCountFlags::TYPE_8,
                SampleCountFlags::TYPE_4,
                SampleCountFlags::TYPE_2,
                SampleCountFlags::TYPE_1
            ].into_iter().find_map(|count| {
                if bits & count != SampleCountFlags::empty() {
                    Some (count)
                } else {
                    None
                }
            }).unwrap()
        };
        dbg!(msaa_count);
        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let transfer_queue = unsafe { std::mem::zeroed() };

        // Init device memory allocator
        let allocator = VulkanAllocator::new(&instance, &card)?;

        // Get presentation info
        let surface_caps = surface.get_physical_device_surface_capabilities(card)?;
        let surface_format = surface.get_physical_device_surface_format(card)?;

        // Init msaa image
        let msaa_image = {
            let create_info = ImageCreateInfo::builder()
                .array_layers(1)
                .extent(Extent3D {
                    width: surface_caps.current_extent.width,
                    height: surface_caps.current_extent.height,
                    depth: 1
                })
                .format(surface_format.format)
                .image_type(ImageType::TYPE_2D)
                .mip_levels(1)
                .samples(msaa_count)
                .usage(ImageUsageFlags::TRANSIENT_ATTACHMENT | ImageUsageFlags::COLOR_ATTACHMENT);
            allocator.allocate_image(&device, &create_info)?
        };
        let msaa_image_view = unsafe {
            let subresource_range = ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let create_info = ImageViewCreateInfo::builder()
                .image(msaa_image)
                .view_type(ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(subresource_range);
            let image_view = device.create_image_view(&create_info, None)?;
            allocator.image_views.borrow_mut().push(image_view);
            image_view
        };

        // Init depth image
        let depth_format = Format::D32_SFLOAT;
        let depth_image = {
            let create_info = ImageCreateInfo::builder()
                .array_layers(1)
                .extent(Extent3D {
                    width: surface_caps.current_extent.width,
                    height: surface_caps.current_extent.height,
                    depth: 1
                })
                .format(depth_format)
                .image_type(ImageType::TYPE_2D)
                .mip_levels(1)
                .samples(msaa_count)
                .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
            allocator.allocate_image(&device, &create_info)?
        };
        let depth_image_view = unsafe {
            let subresource_range = ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let create_info = ImageViewCreateInfo::builder()
                .image(depth_image)
                .view_type(ImageViewType::TYPE_2D)
                .format(depth_format)
                .subresource_range(subresource_range);
            let image_view = device.create_image_view(&create_info, None)?;
            allocator.image_views.borrow_mut().push(image_view);
            image_view
        };

        // Init renderpass
        let render_pass = VulkanInstance::init_renderpass(&device, &surface_format, msaa_count, depth_format)?;

        // Init swapchain
        let swapchain = VulkanSwapchain::new(&surface, &card, &instance, &device, &surface_caps, &surface_format, &render_pass,
            queue_family_indices.to_vec(), &msaa_image_view, &depth_image_view)?;

        // Construct pipeline
        let vertex_input_state = PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: 0
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: 12
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: 24
                }
            ])
            .vertex_binding_descriptions(&[
                VertexInputBindingDescription {
                    binding: 0,
                    stride: 32,
                    input_rate: VertexInputRate::VERTEX
                }
            ]);
        let descriptor_set_layout = {
            let descriptor_bindings = [
                DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(ShaderStageFlags::FRAGMENT)
                    .build()
            ];
            let create_info = DescriptorSetLayoutCreateInfo::builder()
                .bindings(&descriptor_bindings);
            unsafe { device.create_descriptor_set_layout(&create_info, None)? }
        };
        let image_descriptor_set_layout = {
            let bindings = [
                DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::FRAGMENT)
                .build()
            ];
            let create_info = DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings);
            unsafe { device.create_descriptor_set_layout(&create_info, None)? }
        };
        let descriptor_set_layouts = [descriptor_set_layout, image_descriptor_set_layout];
        let pipeline_layout_state = PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&[PushConstantRange {
                offset: 0,
                size: 2 * 4 * 4 * 4, // two mat4x4's
                stage_flags: ShaderStageFlags::VERTEX
            }, PushConstantRange {
                offset: 2 * 4 * 4 * 4,
                size: 4,
                stage_flags: ShaderStageFlags::FRAGMENT
            }])
            .set_layouts(&descriptor_set_layouts);
        let (graphics_pipeline, pipeline_layout) = VulkanPipeline::new_graphics(&device, surface_caps.current_extent, &render_pass, msaa_count,
            *vertex_input_state, *pipeline_layout_state)?;

        // Create command buffers
        let command_pools = VulkanInstance::init_command_pools(&device, vec![0, 0])?;
        let graphics_pool = command_pools[0];
        let transfer_pool = command_pools[1];
        let graphics_command_buffers = VulkanInstance::init_command_buffers(&device, &graphics_pool, swapchain.len() as u32)?;

        Ok (VulkanInstance {
            entry, instance, debug_utils, debug_utils_messenger,
            surface: ManuallyDrop::new(surface),
            device, graphics_queue, transfer_queue, queue_indices: queue_family_indices,
            extent: surface_caps.current_extent, swapchain,
            render_pass,
            descriptor_set_layout, image_descriptor_set_layout, graphics_pipeline, pipeline_layout,
            graphics_pool, transfer_pool,
            graphics_command_buffers,
            allocator, descriptor_pools: vec![].into()
        })
    }

    // TODO organize this better
    pub fn begin_commands(&self) -> Result<CommandBuffer, Box<dyn Error>> {
        let cmdbuf = self.graphics_command_buffers[self.swapchain.index];
        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmdbuf, &begin_info)? };
        let clear_values = [
            ClearValue { color: ClearColorValue { // msaa color attachment
                float32: [0.0, 0.0, 0.00, 1.0]
            } },
            ClearValue::default(), // swapchain attachment
            ClearValue { depth_stencil: ClearDepthStencilValue { // depth attachment
                depth: 1.0,
                stencil: 0
            } }
        ];
        let create_info = RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain.framebuffers[self.swapchain.index])
            .render_area(Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.extent
            })
            .clear_values(&clear_values);
        unsafe {
            self.device.cmd_begin_render_pass(cmdbuf, &create_info, SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(cmdbuf, PipelineBindPoint::GRAPHICS, self.graphics_pipeline); // TODO make caller do this
        }
        Ok (cmdbuf)
    }

    fn init_command_buffers(device: &Device, graphics_pool: &CommandPool, swapchain_size: u32) -> Result<Vec<CommandBuffer>, Box<dyn Error>> {
        let create_info = CommandBufferAllocateInfo::builder()
            .command_pool(*graphics_pool)
            .command_buffer_count(swapchain_size);
        unsafe { device.allocate_command_buffers(&create_info).map_err(|e| Box::new(e) as Box<dyn Error>) }
    }

    fn init_command_pools(device: &Device, queue_families: Vec<u32>) -> Result<Vec<CommandPool>, Box<dyn Error>> {
        queue_families.into_iter().map(|q| {
            let create_info = CommandPoolCreateInfo::builder()
                .queue_family_index(q)
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&create_info, None) }
        }).collect::<Result<Vec<_>, _>>().map_err(|e| Box::new(e) as Box<dyn Error>)
    }

    fn init_renderpass(device: &Device, surface_format: &SurfaceFormatKHR, msaa_count: SampleCountFlags, depth_format: Format) -> Result<RenderPass, Box<dyn Error>> {
        let color_attachment = AttachmentDescription::builder()
            .format(surface_format.format)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .samples(msaa_count);
        let resolve_attachment = AttachmentDescription::builder()
            .format(surface_format.format)
            .load_op(AttachmentLoadOp::DONT_CARE)
            .store_op(AttachmentStoreOp::STORE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .samples(SampleCountFlags::TYPE_1);
        let depth_attachment = AttachmentDescription::builder()
            .format(depth_format)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .samples(msaa_count);
        let attachments = [*color_attachment, *resolve_attachment, *depth_attachment];
        let color_attachment_reference = AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_reference = AttachmentReference::builder()
            .attachment(1)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachment_reference = AttachmentReference::builder()
            .attachment(2)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [*color_attachment_reference];
        let resolve_attachment_references = [*resolve_attachment_reference];
        let subpass = SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .resolve_attachments(&resolve_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference)
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS);
        let subpasses = [*subpass];

        let subpass_dependency = SubpassDependency::builder()
            .src_subpass(SUBPASS_EXTERNAL)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_subpass(0)
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
        let dependencies = [*subpass_dependency];

        let create_info = RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass = unsafe { device.create_render_pass(&create_info, None)? };
        Ok (render_pass)
    }

    // Selects the first available GPU, preferring discrete cards over others
    // TODO allow user choice
    fn select_device(instance: &Instance, surface: &VulkanSurface, layers: &[*const i8]) -> Result<(PhysicalDevice, Device, Vec<u32>), Box<dyn Error>> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        // TODO what if devices len is 0?
        let mut discrete = devices.iter().filter(|d| {
            unsafe { instance.get_physical_device_properties(**d).device_type == PhysicalDeviceType::DISCRETE_GPU }
        });
        let card = discrete.next().unwrap_or(&devices[0]);

        // Request rendering and transfer queues
        // TODO do something smarter than using the same queue for both
        let available = unsafe { instance.get_physical_device_queue_family_properties(*card) };
        let (queue_index, _) = available.iter().enumerate().find(|(i, family)| {
            family.queue_count > 0
            && family.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::TRANSFER)
            && surface.get_physical_device_surface_support(*card, *i as u32)
        }).ok_or("No usable queue family found")?;
        let graphics_queue_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_index as u32)
            .queue_priorities(&[0.5]);
        let queue_infos = [*graphics_queue_info];

        let device_extensions = vec![Swapchain::name().as_ptr()];
        let create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_layer_names(layers);
        let device = unsafe { instance.create_device(*card, &create_info, None) }.map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok ((*card, device, vec![queue_index as u32]))
    }
}
