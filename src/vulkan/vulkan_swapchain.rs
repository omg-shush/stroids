use std::error::Error;

use ash::extensions::khr::Swapchain;
use ash::{Device, Instance};
use ash::vk::{Queue, RenderPass, PhysicalDevice, Semaphore, Fence, SwapchainKHR, ImageView, Framebuffer, SemaphoreCreateInfo, FenceCreateInfo, FenceCreateFlags, Extent2D, SurfaceCapabilitiesKHR, SurfaceFormatKHR, PresentModeKHR, SwapchainCreateInfoKHR, ImageUsageFlags, SharingMode, CompositeAlphaFlagsKHR, ImageSubresourceRange, ImageAspectFlags, ImageViewCreateInfo, ImageViewType, FramebufferCreateInfo, PresentInfoKHR};

use super::vulkan_surface::VulkanSurface;

pub struct VulkanSwapchain {
    loader: Swapchain,
    pub swapchain: SwapchainKHR,
    image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
    pub semaphore_image_available: Vec<Semaphore>,
    pub semaphore_rendering_finished: Vec<Semaphore>,
    pub fence_resources_ready: Vec<Fence>,
    pub index: usize, // Current image index in swapchain, ranging 0 to (image_views.len() - 1)
}

impl VulkanSwapchain {
    pub fn new(surface: &VulkanSurface, card: &PhysicalDevice, instance: &Instance, device: &Device,
            surface_caps: &SurfaceCapabilitiesKHR, surface_format: &SurfaceFormatKHR, render_pass: &RenderPass, queue_family_indices: Vec<u32>,
            msaa_image_view: &ImageView, depth_image_view: &ImageView)
            -> Result<VulkanSwapchain, Box<dyn Error>> {
        // Construct swapchain images
        let (loader, swapchain, image_views) = VulkanSwapchain::init_swapchain(
            &surface, &card, &instance, &device, &surface_caps, surface_format, &queue_family_indices)?;
        let last_index = image_views.len() - 1;

        // Connect renderpass and swapchain
        let framebuffers = VulkanSwapchain::init_framebuffers(&device, &render_pass, &image_views, surface_caps.current_extent, msaa_image_view, depth_image_view)?;

        // Init synchronization
        let semaphore_create_info = SemaphoreCreateInfo::builder();
        let fence_create_info = FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED);
        let semaphore_image_available = unsafe { vec![0; image_views.len()].into_iter().map(|_| device.create_semaphore(&semaphore_create_info, None)).collect::<Result<Vec<_>, _>>()? };
        let semaphore_rendering_finished = unsafe { vec![0; image_views.len()].into_iter().map(|_| device.create_semaphore(&semaphore_create_info, None)).collect::<Result<Vec<_>, _>>()? };
        let fence_draw_ready = unsafe { vec![0; image_views.len()].into_iter().map(|_| device.create_fence(&fence_create_info, None)).collect::<Result<Vec<_>, _>>()? };
    
        Ok (VulkanSwapchain {
            loader, swapchain, image_views, framebuffers, semaphore_image_available, semaphore_rendering_finished, fence_resources_ready: fence_draw_ready,
            index: last_index // Init to last index, so next index will be 0
        })
    }

    pub fn drop(&self, device: &Device) {
        unsafe {
            self.image_views.iter().for_each(|image_view| device.destroy_image_view(*image_view, None));
            self.loader.destroy_swapchain(self.swapchain, None);
            self.semaphore_image_available.iter().for_each(|semaphore| device.destroy_semaphore(*semaphore, None));
            self.semaphore_rendering_finished.iter().for_each(|semaphore| device.destroy_semaphore(*semaphore, None));
            self.fence_resources_ready.iter().for_each(|fence| device.destroy_fence(*fence, None));
            self.framebuffers.iter().for_each(|framebuffer| device.destroy_framebuffer(*framebuffer, None));
        }
    }

    // Acquires the next swapchain image, updating the current image index and waiting for the image to be ready
    pub fn acquire(&mut self, device: &Device) {
        let next_index = (self.index + 1) % self.len();
        let (index, _suboptimal) = unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.semaphore_image_available[next_index],
                Fence::null()).expect("Failed to acquire image") // TODO
        };
        self.index = index as usize;

        // Wait till the GPU is done with the image
        unsafe {
            device.wait_for_fences(&[
                self.fence_resources_ready[self.index]
            ], true, u64::MAX).expect("Waiting for fences");
            // Now guaranteed that this image's command buffer has been consumed and reset
            device.reset_fences(&[
                self.fence_resources_ready[self.index]
            ]).expect("Resetting fences");
        }
    }

    pub fn present(&self, present_queue: &Queue) {
        // Present next image on screen
        let swapchains = [self.swapchain];
        let indices = [self.index as u32];
        let semaphores_finished = [self.semaphore_rendering_finished[self.index]];
        let create_info = PresentInfoKHR::builder()
            .wait_semaphores(&semaphores_finished)
            .swapchains(&swapchains)
            .image_indices(&indices);
        unsafe {
            self.loader.queue_present(
                *present_queue,
                &create_info)
                .expect("Presenting next image from swapchain");
        }
    }

    // Returns number of images in the swapchain
    pub fn len(&self) -> usize {
        self.image_views.len()
    }

    fn init_swapchain(surface: &VulkanSurface, card: &PhysicalDevice, instance: &Instance, device: &Device,
            surface_caps: &SurfaceCapabilitiesKHR, surface_format: &SurfaceFormatKHR, queues: &Vec<u32>)
            -> Result<(Swapchain, SwapchainKHR, Vec<ImageView>), Box<dyn Error>> {
        let surface_presents = surface.get_physical_device_surface_present_modes(*card)?;

        let present_mode = if surface_presents.contains(&PresentModeKHR::FIFO_RELAXED) {
            PresentModeKHR::FIFO_RELAXED
        } else {
            PresentModeKHR::FIFO
        };
        let create_info = SwapchainCreateInfoKHR::builder()
            .surface(surface.surface)
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

    fn init_framebuffers(device: &Device, render_pass: &RenderPass, swapchain_image_views: &Vec<ImageView>, extent: Extent2D, msaa_image_view: &ImageView, depth_image_view: &ImageView)
            -> Result<Vec<Framebuffer>, Box<dyn Error>> {
        swapchain_image_views.iter().map(|swapchain_image_view| {
            let attachments = [*msaa_image_view, *swapchain_image_view, *depth_image_view];
            let create_info = FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&create_info, None) }
        }).collect::<Result<Vec<_>, _>>().map_err(|e| Box::new(e) as Box<dyn Error>)
    }
}
