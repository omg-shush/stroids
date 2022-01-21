use std::cell::RefCell;
use std::error::Error;
use std::ffi::{CString, c_void, NulError};
use std::mem::ManuallyDrop;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain, Win32Surface, XlibSurface};
use ash::{vk, Entry, Instance, Device};
use ash::vk::{Format, Pipeline, RenderPass, Queue, PhysicalDevice, PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, QueueFlags, SurfaceFormatKHR, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, ImageLayout, SampleCountFlags, AttachmentReference, SubpassDescription, PipelineBindPoint, SubpassDependency, SUBPASS_EXTERNAL, PipelineStageFlags, AccessFlags, RenderPassCreateInfo, Extent2D, ShaderModuleCreateInfo, PipelineShaderStageCreateInfo, ShaderStageFlags, PipelineVertexInputStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PrimitiveTopology, Viewport, Rect2D, Offset2D, PipelineViewportStateCreateInfo, PipelineRasterizationStateCreateInfo, FrontFace, CullModeFlags, PolygonMode, PipelineMultisampleStateCreateInfo, PipelineColorBlendAttachmentState, BlendFactor, BlendOp, ColorComponentFlags, PipelineColorBlendStateCreateInfo, PipelineLayoutCreateInfo, GraphicsPipelineCreateInfo, PipelineCache, PipelineLayout, CommandPool, CommandPoolCreateInfo, CommandPoolCreateFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferUsageFlags, ClearValue, ClearColorValue, RenderPassBeginInfo, SubpassContents, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, PushConstantRange, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DescriptorSetLayoutCreateInfo, DescriptorPool};
use vk_shader_macros::include_glsl;
use winit::window::Window;

use super::vulkan_swapchain::VulkanSwapchain;
use super::vulkan_allocator::VulkanAllocator;
use super::vulkan_surface::VulkanSurface;

pub struct VulkanInstance {
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    surface: ManuallyDrop<VulkanSurface>,
    pub device: Device,
    pub graphics_queue: Queue,
    transfer_queue: Queue,
    extent: Extent2D,
    pub swapchain: ManuallyDrop<VulkanSwapchain>,
    render_pass: RenderPass,
    pub descriptor_set_layout: DescriptorSetLayout,
    graphics_pipeline: Pipeline,
    pub pipeline_layout: PipelineLayout,
    graphics_pool: CommandPool,
    transfer_pool: CommandPool,
    pub graphics_command_buffers: Vec<CommandBuffer>,
    pub allocator: ManuallyDrop<VulkanAllocator>,
    pub descriptor_pools: RefCell<Vec<DescriptorPool>>
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait for device idle");

            self.device.destroy_command_pool(self.graphics_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_pools.borrow().iter().for_each(|pool| self.device.destroy_descriptor_pool(*pool, None));
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            ManuallyDrop::drop(&mut self.swapchain);
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
        let entry = unsafe { Entry::new() }?;

        // Describe application
        let app_name = CString::new("'Stroids")?;
        let engine_name = CString::new("Astral Engine")?;
        let app_info = ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 1, 0))
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
        let surface = VulkanSurface::new(&entry, &instance, &window)?;

        // Init device & queues
        let (card, device, queue_family_indices) = VulkanInstance::select_device(&instance,&surface, &layers)?;
        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let transfer_queue = graphics_queue.clone();

        // Get presentation info
        let surface_caps = surface.get_physical_device_surface_capabilities(card)?;
        let surface_format = surface.get_physical_device_surface_format(card)?;

        // Init renderpass
        let render_pass = VulkanInstance::init_renderpass(&device, &surface_format)?;

        // Init swapchain
        let swapchain = VulkanSwapchain::new(&surface, &card, &instance, &device, &surface_caps, &surface_format, &render_pass,
            queue_family_indices)?;

        // Init device memory allocator
        let allocator = VulkanAllocator::new(&instance, &card, &device)?;

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
                    format: Format::R32_SFLOAT,
                    offset: 12
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32_SFLOAT,
                    offset: 16
                },
                VertexInputAttributeDescription {
                    location: 3,
                    binding: 0,
                    format: Format::R32_SFLOAT,
                    offset: 20
                },
            ])
            .vertex_binding_descriptions(&[VertexInputBindingDescription {
                binding: 0,
                stride: 24,
                input_rate: VertexInputRate::VERTEX
            }]);
        let descriptor_bindings = [DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::FRAGMENT)
            .build()
        ];
        let descriptor_set_layout = {
            let create_info = DescriptorSetLayoutCreateInfo::builder()
                .bindings(&descriptor_bindings);
            unsafe { device.create_descriptor_set_layout(&create_info, None)? }
        };
        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout_state = PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&[PushConstantRange {
                offset: 0,
                size: 4,
                stage_flags: ShaderStageFlags::VERTEX
            }])
            .set_layouts(&descriptor_set_layouts);
        let (graphics_pipeline, pipeline_layout) = VulkanInstance::init_pipeline(&device, surface_caps.current_extent, &render_pass,
            *vertex_input_state, *pipeline_layout_state)?;

        // Create command buffers
        let command_pools = VulkanInstance::init_command_pools(&device, vec![0, 0])?;
        let graphics_pool = command_pools[0];
        let transfer_pool = command_pools[1];
        let graphics_command_buffers = VulkanInstance::init_command_buffers(&device, &graphics_pool, swapchain.len() as u32)?;

        Ok (VulkanInstance {
            instance, debug_utils, debug_utils_messenger,
            surface: ManuallyDrop::new(surface),
            device, graphics_queue, transfer_queue,
            extent: surface_caps.current_extent, swapchain: ManuallyDrop::new(swapchain),
            render_pass,
            descriptor_set_layout, graphics_pipeline, pipeline_layout,
            graphics_pool, transfer_pool,
            graphics_command_buffers,
            allocator: ManuallyDrop::new(allocator), descriptor_pools: vec![].into()
        })
    }

    // TODO organize this better
    pub fn begin_commands(&self) -> Result<CommandBuffer, Box<dyn Error>> {
        let cmdbuf = self.graphics_command_buffers[self.swapchain.index];
        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmdbuf, &begin_info)? };
        let clear_values = [ClearValue {
            color: ClearColorValue {
                float32: [0.0, 0.0, 0.08, 1.0]
            }
        }];
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

    fn init_pipeline(device: &Device, extent: Extent2D, render_pass: &RenderPass,
        vertex_input_state: PipelineVertexInputStateCreateInfo,
        pipeline_layout_state: PipelineLayoutCreateInfo) -> Result<(Pipeline, PipelineLayout), Box<dyn Error>> {
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
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_state, None)? };
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
