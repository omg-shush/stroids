use std::error::Error;
use std::ffi::{CString, c_void, NulError};

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface};
use ash::{vk, Entry, Instance, Device};
use ash::vk::{PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, QueueFlags, Win32SurfaceCreateInfoKHR, Win32SurfaceCreateFlagsKHR, SurfaceKHR};
use winit::platform::windows::WindowExtWindows;
use winit::window::Window;

pub struct VulkanInstance {
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    surface_loader: Surface,
    surface: SurfaceKHR,
    device: Device
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
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
        let device = VulkanInstance::select_device(&instance, &surface_loader, &surface)?;
        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let draw_queue = graphics_queue.clone();

        Ok (VulkanInstance { instance, debug_utils, debug_utils_messenger, surface_loader, surface, device })
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
    fn select_device(instance: &Instance, surface_loader: &Surface, surface: &SurfaceKHR) -> Result<Device, Box<dyn Error>> {
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

        let create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos);
        let device = unsafe { instance.create_device(*selected, &create_info, None) }.map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Ok (device)
    }
}
