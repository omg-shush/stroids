// Platform-specific surface initialization

use std::error::Error;

use ash::{Entry, Instance};
use ash::vk::{self, SurfaceFormatKHR, PresentModeKHR};
use ash::vk::{PhysicalDevice, SurfaceKHR, SurfaceCapabilitiesKHR};
use ash::extensions::khr::Surface;
use winit::window::Window;

pub struct VulkanSurface {
    loader: Surface,
    pub surface: SurfaceKHR
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

impl VulkanSurface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> Result<VulkanSurface, Box<dyn Error>> {
        Ok (VulkanSurface {
            loader: Surface::new(&entry, &instance),
            surface: create_surface(entry, instance, window)?
        })
    }

    pub fn get_physical_device_surface_support(&self, card: PhysicalDevice, i: u32) -> bool {
        unsafe { self.loader.get_physical_device_surface_support(card, i, self.surface) }.unwrap_or(false)
    }

    pub fn get_physical_device_surface_capabilities(&self, card: PhysicalDevice) -> Result<SurfaceCapabilitiesKHR, vk::Result> {
        unsafe { self.loader.get_physical_device_surface_capabilities(card, self.surface) }
    }

    pub fn get_physical_device_surface_format(&self, card: PhysicalDevice) -> Result<SurfaceFormatKHR, vk::Result> {
        let formats = unsafe { self.loader.get_physical_device_surface_formats(card, self.surface) }?;
        // TODO smarter selection of format
        Ok (formats[0])
    }

    pub fn get_physical_device_surface_present_modes(&self, card: PhysicalDevice) -> Result<Vec<PresentModeKHR>, vk::Result> {
        unsafe { self.loader.get_physical_device_surface_present_modes(card, self.surface) }
    }
}

#[cfg(windows)]
fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> Result<SurfaceKHR, Box<dyn Error>> {
    use ash::vk::Win32SurfaceCreateInfoKHR;
    use winit::platform::windows::WindowExtWindows;
    use ash::extensions::khr::Win32Surface;

    let win32_surface_loader = Win32Surface::new(entry, instance);
    let create_info = Win32SurfaceCreateInfoKHR::builder()
        .hinstance(window.hinstance())
        .hwnd(window.hwnd());
    let surface = unsafe { win32_surface_loader.create_win32_surface(&create_info, None) }?;
    Ok (surface)
}

#[cfg(unix)]
fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> Result<SurfaceKHR, Box<dyn Error>> {
    use std::ffi::c_void;
    
    use winit::platform::unix::WindowExtUnix;
    use ash::{extensions::khr::XlibSurface, vk::{XlibSurfaceCreateInfoKHR, Window}};

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    let create_info = XlibSurfaceCreateInfoKHR::builder()
        .dpy(window.xlib_display().ok_or("Xlib display not found")? as *mut *const c_void)
        .window(window.xlib_window().ok_or("Xlib window not found")? as Window);
    let surface = unsafe { xlib_surface_loader.create_xlib_surface(&create_info, None) }?;
    Ok (surface)
}
