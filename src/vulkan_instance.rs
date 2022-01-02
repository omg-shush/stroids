use std::error::Error;
use std::ffi::{CString, c_void, NulError};

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface};
use ash::prelude::VkResult;
use ash::{vk, Entry, Instance, Device};
use ash::vk::{PhysicalDeviceType, DeviceCreateInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32, DeviceQueueCreateInfoBuilder, DeviceQueueCreateInfo, ApplicationInfo, InstanceCreateInfo, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT};

pub struct VulkanInstance {
    instance: Instance,
    debug_utils: DebugUtils,
    debug_utils_messenger: DebugUtilsMessengerEXT,
    device: Device
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
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
    pub fn new() -> Result<VulkanInstance, Box<dyn Error>> {
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

        let device = VulkanInstance::select_device(&instance)?;

        Ok (VulkanInstance { instance, debug_utils, debug_utils_messenger, device })
    }

    // Selects the first available GPU, preferring discrete cards over others
    // TODO allow user choice
    fn select_device(instance: &Instance) -> VkResult<Device> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        // TODO what if devices len is 0?
        let mut discrete = devices.iter().filter(|d| {
            unsafe { instance.get_physical_device_properties(**d).device_type == PhysicalDeviceType::DISCRETE_GPU }
        });
        let selected = discrete.next().unwrap_or(&devices[0]);
        // let queues = DeviceQueueCreateInfo::builder()
        let create_info = DeviceCreateInfo::builder();
        unsafe { instance.create_device(*selected, &create_info, None) }
    }
}
