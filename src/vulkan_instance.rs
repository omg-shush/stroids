use std::error::Error;
use std::ffi::CString;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface, XlibSurface};
use ash::prelude::VkResult;
use ash::{vk, Entry, Instance, Device};
use ash::vk::{PhysicalDeviceType, DeviceCreateInfo};

pub struct VulkanInstance {
    entry: Entry,
    instance: Instance,
    device: Device
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl VulkanInstance {
    pub fn new() -> Result<VulkanInstance, Box<dyn Error>> {
        let entry = unsafe { Entry::new() }?;
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_application_name: CString::new("'Stroids")?.as_ptr(),
            p_engine_name: CString::new("Astral Engine")?.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            ..Default::default()
        };
        let supported_extensions: Vec<CString> = entry.enumerate_instance_extension_properties()?.into_iter().map(|ep| {
            CString::new(ep.extension_name.map(|i| i as u8).into_iter().take_while(|c| *c != 0).collect::<Vec<_>>())
        }).collect::<Result<Vec<CString>, _>>()?;
        let desired_extensions = vec![
            Surface::name().to_owned(),
            Win32Surface::name().to_owned(), // TODO cross platform
            DebugUtils::name().to_owned()
        ];
        let extensions = desired_extensions.iter().filter(|e| supported_extensions.contains(e)).map(|e| e.as_ptr()).collect::<Vec<_>>();
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let device = VulkanInstance::select_device(&instance)?;
        Ok (VulkanInstance { entry, instance, device })
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
        let create_info = DeviceCreateInfo::builder().build();
        unsafe { instance.create_device(*selected, &create_info, None) }
    }
}
