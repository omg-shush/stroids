use std::error::Error;

use ash::{vk, Entry, Instance};

pub struct VulkanInstance {
    entry: Entry,
    instance: Instance
}

impl VulkanInstance {
    pub fn new() -> Result<VulkanInstance, Box<dyn Error>> {
        let entry = unsafe { Entry::new() }?;
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok (VulkanInstance { entry, instance })
    }
}
