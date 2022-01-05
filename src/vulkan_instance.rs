use std::error::Error;
use std::ffi::{CString, c_void, NulError};

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface, Swapchain};
use ash::{vk, Entry, Instance, Device};
use ash::vk::{Queue, ImageView, PhysicalDevice, PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, QueueFlags, Win32SurfaceCreateInfoKHR, SurfaceKHR, SwapchainKHR, SwapchainCreateInfoKHR, ImageUsageFlags, SharingMode, CompositeAlphaFlagsKHR, PresentModeKHR, ImageViewCreateInfo, ImageViewType, ImageSubresourceRange, ImageAspectFlags};
use winit::platform::windows::WindowExtWindows;
use winit::window::Window;

pub struct VulkanInstance {
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    surface_loader: Surface,
    surface: SurfaceKHR,
    device: Device,
    graphics_queue: Queue,
    transfer_queue: Queue,
    swapchain_loader: Swapchain,
    swapchain: SwapchainKHR,
    swapchain_image_views: Vec<ImageView>,

}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
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
        let (swapchain_loader, swapchain, swapchain_image_views) = VulkanInstance::init_swapchain(&surface_loader, &surface, &card, &instance, &device, &queue_family_indices)?;

        Ok (VulkanInstance { instance, debug_utils, debug_utils_messenger, surface_loader, surface, device, graphics_queue, transfer_queue, swapchain_loader, swapchain, swapchain_image_views })
    }

    fn init_swapchain(surface_loader: &Surface, surface: &SurfaceKHR, card: &PhysicalDevice, instance: &Instance, device: &Device, queues: &Vec<u32>) -> Result<(Swapchain, SwapchainKHR, Vec<ImageView>), Box<dyn Error>> {
        let surface_caps = unsafe { surface_loader.get_physical_device_surface_capabilities(*card, *surface) }?;
        let surface_presents = unsafe { surface_loader.get_physical_device_surface_present_modes(*card, *surface) }?;
        let surface_formats = unsafe { surface_loader.get_physical_device_surface_formats(*card, *surface) }?;

        // TODO smarter selection of format
        let surface_format = surface_formats.get(0).ok_or("No surface formats available")?;
        let present_mode = if surface_presents.contains(&PresentModeKHR::FIFO_RELAXED) {
            PresentModeKHR::FIFO_RELAXED
        } else {
            PresentModeKHR::FIFO
        };
        let create_info = SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(3.clamp(surface_caps.min_image_count, surface_caps.max_image_count))
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
