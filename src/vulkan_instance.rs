use std::error::Error;
use std::ffi::{CString, c_void, NulError};

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface, Swapchain};
use ash::{vk, Entry, Instance, Device};
use ash::vk::{Fence, Pipeline, RenderPass, Queue, ImageView, PhysicalDevice, PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, QueueFlags, Win32SurfaceCreateInfoKHR, SurfaceKHR, SwapchainKHR, SwapchainCreateInfoKHR, ImageUsageFlags, SharingMode, CompositeAlphaFlagsKHR, PresentModeKHR, ImageViewCreateInfo, ImageViewType, ImageSubresourceRange, ImageAspectFlags, SurfaceFormatKHR, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, ImageLayout, SampleCountFlags, AttachmentReference, SubpassDescription, PipelineBindPoint, SubpassDependency, SUBPASS_EXTERNAL, PipelineStageFlags, AccessFlags, RenderPassCreateInfo, Framebuffer, Extent2D, SurfaceCapabilitiesKHR, FramebufferCreateInfo, ShaderModuleCreateInfo, PipelineShaderStageCreateInfo, ShaderStageFlags, PipelineVertexInputStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PrimitiveTopology, Viewport, Rect2D, Offset2D, PipelineViewportStateCreateInfo, PipelineRasterizationStateCreateInfo, FrontFace, CullModeFlags, PolygonMode, PipelineMultisampleStateCreateInfo, PipelineColorBlendAttachmentState, BlendFactor, BlendOp, ColorComponentFlags, PipelineColorBlendStateCreateInfo, PipelineLayoutCreateInfo, GraphicsPipelineCreateInfo, PipelineCache, PipelineLayout, CommandPool, CommandPoolCreateInfo, CommandPoolCreateFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferUsageFlags, ClearValue, ClearColorValue, RenderPassBeginInfo, SubpassContents, SemaphoreCreateInfo, Semaphore, FenceCreateInfo, FenceCreateFlags};
use vk_shader_macros::include_glsl;
use winit::platform::windows::WindowExtWindows;
use winit::window::Window;

pub struct VulkanInstance {
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    surface_loader: Surface,
    surface: SurfaceKHR,
    pub device: Device,
    pub graphics_queue: Queue,
    transfer_queue: Queue,
    pub swapchain_loader: Swapchain,
    pub swapchain: SwapchainKHR,
    pub swapchain_image_views: Vec<ImageView>,
    pub swapchain_ptr: usize,
    render_pass: RenderPass,
    framebuffers: Vec<Framebuffer>,
    graphics_pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    graphics_pool: CommandPool,
    transfer_pool: CommandPool,
    pub graphics_command_buffers: Vec<CommandBuffer>,
    pub semaphore_image_available: Vec<Semaphore>,
    pub semaphore_rendering_finished: Vec<Semaphore>,
    pub fence_draw_ready: Vec<Fence>
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait for device idle");

            self.device.destroy_command_pool(self.graphics_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.semaphore_image_available.iter().for_each(|semaphore| self.device.destroy_semaphore(*semaphore, None));
            self.semaphore_rendering_finished.iter().for_each(|semaphore| self.device.destroy_semaphore(*semaphore, None));
            self.fence_draw_ready.iter().for_each(|fence| self.device.destroy_fence(*fence, None));
            self.framebuffers.iter().for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views.iter().for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
            self.debug_utils.destroy_debug_utils_messenger(self.debug_utils_messenger, None)
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
        let entry = unsafe { Entry::new() }?;

        // Describe application
        let app_name = CString::new("'Stroids")?;
        let engine_name = CString::new("Astral Engine")?;
        let app_info = ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 0, 0))
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
            Win32Surface::name().to_owned(), // TODO cross platform
            DebugUtils::name().to_owned()
        ];
        let extensions = desired_extensions.iter().filter(|e| supported_extensions.contains(e)).map(|e| e.as_ptr()).collect::<Vec<_>>();
        let layer_names = vec!["VK_LAYER_KHRONOS_validation"].into_iter().map(|s| CString::new(s)).collect::<Result<Vec<_>, NulError>>()?;
        let layers = layer_names.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        
        // Construct instance
        let mut debug_create_info = DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(DebugUtilsMessageSeverityFlagsEXT::WARNING | DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(DebugUtilsMessageTypeFlagsEXT::all())
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
        let (surface_loader, surface) = VulkanInstance::init_win32(&entry, &instance, &window)?;

        // Init device & queues
        let (card, device, queue_family_indices) = VulkanInstance::select_device(&instance, &surface_loader, &surface, &layers)?;
        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let transfer_queue = graphics_queue.clone();

        // Init swapchain
        let surface_caps = unsafe { surface_loader.get_physical_device_surface_capabilities(card, surface) }?;
        let surface_formats = unsafe { surface_loader.get_physical_device_surface_formats(card, surface) }?;
        // TODO smarter selection of format
        let surface_format = surface_formats.get(0).ok_or("No surface formats available")?;
        let (swapchain_loader, swapchain, swapchain_image_views) = VulkanInstance::init_swapchain(&surface_loader, &surface, &card, &instance, &device, &surface_caps, surface_format, &queue_family_indices)?;

        // Init renderpass
        let render_pass = VulkanInstance::init_renderpass(&device, surface_format)?;
        let framebuffers = VulkanInstance::init_framebuffers(&device, &render_pass, &swapchain_image_views, surface_caps.current_extent)?;

        // Construct pipeline
        let (graphics_pipeline, pipeline_layout) = VulkanInstance::init_pipeline(&device, surface_caps.current_extent, &render_pass)?;

        // Create command buffers
        let command_pools = VulkanInstance::init_command_pools(&device, vec![0, 0])?;
        let graphics_pool = command_pools[0];
        let transfer_pool = command_pools[1];
        let graphics_command_buffers = VulkanInstance::init_command_buffers(&device, &graphics_pool, swapchain_image_views.len() as u32)?;

        // Add rendering commands to graphics command buffers
        VulkanInstance::write_graphics_commands(&device, &render_pass, &framebuffers, surface_caps.current_extent, &graphics_pipeline, &graphics_command_buffers)?;

        // Init synchronization
        let semaphore_create_info = SemaphoreCreateInfo::builder();
        let fence_create_info = FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED);
        let semaphore_image_available = unsafe { vec![0; swapchain_image_views.len()].into_iter().map(|_| device.create_semaphore(&semaphore_create_info, None)).collect::<Result<Vec<_>, _>>()? };
        let semaphore_rendering_finished = unsafe { vec![0; swapchain_image_views.len()].into_iter().map(|_| device.create_semaphore(&semaphore_create_info, None)).collect::<Result<Vec<_>, _>>()? };
        let fence_draw_ready = unsafe { vec![0; swapchain_image_views.len()].into_iter().map(|_| device.create_fence(&fence_create_info, None)).collect::<Result<Vec<_>, _>>()? };

        Ok (VulkanInstance {
            instance, debug_utils, debug_utils_messenger,
            surface_loader, surface,
            device, graphics_queue, transfer_queue,
            swapchain_loader, swapchain, swapchain_image_views, swapchain_ptr: 0,
            render_pass, framebuffers,
            graphics_pipeline, pipeline_layout,
            graphics_pool, transfer_pool,
            graphics_command_buffers,
            semaphore_image_available, semaphore_rendering_finished,
            fence_draw_ready
        })
    }

    fn write_graphics_commands(device: &Device, render_pass: &RenderPass, framebuffers: &Vec<Framebuffer>, extent: Extent2D, pipeline: &Pipeline, command_buffers: &Vec<CommandBuffer>) -> Result<(), Box<dyn Error>> {
        for (i, &buf) in command_buffers.iter().enumerate() {
            let begin_info = CommandBufferBeginInfo::builder()
                .flags(CommandBufferUsageFlags::empty()); // TODO eventually one time submit
            unsafe { device.begin_command_buffer(buf, &begin_info)? };
            let clear_values = [ClearValue {
                color: ClearColorValue {
                    float32: [0.0, 0.0, 0.08, 1.0]
                }
            }];
            let create_info = RenderPassBeginInfo::builder()
                .render_pass(*render_pass)
                .framebuffer(framebuffers[i])
                .render_area(Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent
                })
                .clear_values(&clear_values);
            unsafe {
                device.cmd_begin_render_pass(buf, &create_info, SubpassContents::INLINE);
                device.cmd_bind_pipeline(buf, PipelineBindPoint::GRAPHICS, *pipeline);
                device.cmd_draw(buf, 1, 1, 0, 0);
                device.cmd_end_render_pass(buf);
                device.end_command_buffer(buf)?;
            }
        }
        Ok (())
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

    fn init_pipeline(device: &Device, extent: Extent2D, render_pass: &RenderPass) -> Result<(Pipeline, PipelineLayout), Box<dyn Error>> {
        let vertex_module = {
            let create_info = ShaderModuleCreateInfo::builder()
                .code(include_glsl!("./shaders/shader.vert"));
            unsafe { device.create_shader_module(&create_info, None)? }
        };
        let fragment_module = {
            let create_info = ShaderModuleCreateInfo::builder()
                .code(include_glsl!("./shaders/shader.frag"));
            unsafe { device.create_shader_module(&create_info, None)? }
        };
        let shader_entry = CString::new("main")?;
        let vertex_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&shader_entry);
        let fragment_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&shader_entry);
        let vertex_input_state = PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_state = PipelineInputAssemblyStateCreateInfo::builder()
            .topology(PrimitiveTopology::POINT_LIST);
        let viewports = [ Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0
        } ];
        let scissors = [ Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent
        } ];
        let viewport_state = PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization_state = PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(FrontFace::CLOCKWISE)
            .cull_mode(CullModeFlags::NONE)
            .polygon_mode(PolygonMode::FILL);
        let multisample_state = PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(SampleCountFlags::TYPE_1);
        let color_blend_attachments = [ PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(BlendOp::ADD)
            .color_write_mask(ColorComponentFlags::all())
            .build()
        ];
        let color_blend_state = PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments);
        let pipeline_layout = {
            let create_info = PipelineLayoutCreateInfo::builder();
            unsafe { device.create_pipeline_layout(&create_info, None)? }
        };
        let graphics_pipeline = {
            let create_infos = [ GraphicsPipelineCreateInfo::builder()
                .stages(&[*vertex_stage, *fragment_stage])
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .layout(pipeline_layout)
                .render_pass(*render_pass)
                .subpass(0)
                .build()
            ];
            unsafe { device.create_graphics_pipelines(PipelineCache::null(), &create_infos, None).map_err(|t| t.1)? }
        }[0];

        // Clean up shader modules
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }
        
        Ok ((graphics_pipeline, pipeline_layout))
    }

    fn init_framebuffers(device: &Device, render_pass: &RenderPass, swapchain_image_views: &Vec<ImageView>, extent: Extent2D) -> Result<Vec<Framebuffer>, Box<dyn Error>> {
        swapchain_image_views.iter().map(|image_view| {
            let attachments = [*image_view];
            let create_info = FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&create_info, None) }
        }).collect::<Result<Vec<_>, _>>().map_err(|e| Box::new(e) as Box<dyn Error>)
    }

    fn init_renderpass(device: &Device, surface_format: &SurfaceFormatKHR) -> Result<RenderPass, Box<dyn Error>> {
        let color_attachment = AttachmentDescription::builder()
            .format(surface_format.format)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .samples(SampleCountFlags::TYPE_1);
        let attachments = [*color_attachment];
        let color_attachment_reference = AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [*color_attachment_reference];
        let subpass = SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS);
        let subpasses = [*subpass];

        let subpass_dependency = SubpassDependency::builder()
            .src_subpass(SUBPASS_EXTERNAL)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE);
        let dependencies = [*subpass_dependency];

        let create_info = RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass = unsafe { device.create_render_pass(&create_info, None)? };
        Ok (render_pass)
    }

    fn init_swapchain(surface_loader: &Surface, surface: &SurfaceKHR, card: &PhysicalDevice, instance: &Instance, device: &Device, surface_caps: &SurfaceCapabilitiesKHR, surface_format: &SurfaceFormatKHR, queues: &Vec<u32>) -> Result<(Swapchain, SwapchainKHR, Vec<ImageView>), Box<dyn Error>> {
        let surface_presents = unsafe { surface_loader.get_physical_device_surface_present_modes(*card, *surface) }?;

        let present_mode = if surface_presents.contains(&PresentModeKHR::FIFO_RELAXED) {
            PresentModeKHR::FIFO_RELAXED
        } else {
            PresentModeKHR::FIFO
        };
        let create_info = SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(3.clamp(surface_caps.min_image_count, if surface_caps.max_image_count == 0 { u32::MAX } else { surface_caps.max_image_count }))
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_caps.current_extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .queue_family_indices(&queues)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode);

        let swapchain_loader = Swapchain::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };
        
        // Create image views
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let swapchain_image_views = swapchain_images.iter().map(|image| {
            let subresource_range = ImageSubresourceRange::builder()
                .aspect_mask(ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let create_info = ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(*subresource_range);
            unsafe { device.create_image_view(&create_info, None) }
        }).collect::<Result<Vec<_>, _>>()?;
        
        Ok ((swapchain_loader, swapchain, swapchain_image_views))
    }

    fn init_win32(entry: &Entry, instance: &Instance, window: &Window) -> Result<(Surface, SurfaceKHR), Box<dyn Error>> {
        let win32_surface_loader = Win32Surface::new(entry, instance);
        let create_info = Win32SurfaceCreateInfoKHR::builder()
            .hinstance(window.hinstance())
            .hwnd(window.hwnd());
        let surface = unsafe { win32_surface_loader.create_win32_surface(&create_info, None) }?;
        let surface_loader = Surface::new(&entry, &instance);
        Ok ((surface_loader, surface))
    }

    // Selects the first available GPU, preferring discrete cards over others
    // TODO allow user choice
    fn select_device(instance: &Instance, surface_loader: &Surface, surface: &SurfaceKHR, layers: &[*const i8]) -> Result<(PhysicalDevice, Device, Vec<u32>), Box<dyn Error>> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        // TODO what if devices len is 0?
        let mut discrete = devices.iter().filter(|d| {
            unsafe { instance.get_physical_device_properties(**d).device_type == PhysicalDeviceType::DISCRETE_GPU }
        });
        let selected = discrete.next().unwrap_or(&devices[0]);

        // Request rendering and transfer queues
        // TODO do something smarter than using the same queue for both
        let available = unsafe { instance.get_physical_device_queue_family_properties(*selected) };
        let (queue_index, _) = available.iter().enumerate().find(|(i, family)| {
            family.queue_count > 0
            && family.queue_flags.contains(QueueFlags::GRAPHICS | QueueFlags::TRANSFER)
            && unsafe { surface_loader.get_physical_device_surface_support(*selected, *i as u32, *surface).unwrap_or(false) }
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
        let device = unsafe { instance.create_device(*selected, &create_info, None) }.map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok ((*selected, device, vec![queue_index as u32]))
    }
}
