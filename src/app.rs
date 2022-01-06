use std::error::Error;
use std::time::{Instant, Duration};

use winit::dpi::LogicalSize;
use winit::event::{WindowEvent, Event, KeyboardInput, ElementState, VirtualKeyCode};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use ash::vk::{Fence, PipelineStageFlags, SubmitInfo, PresentInfoKHR};

use crate::vulkan_instance::VulkanInstance;

pub struct App {
    event_loop: EventLoop<()>,
    pub window: Window
}

impl App {
    pub fn new() -> Result<App, Box<dyn Error>> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
                .with_title("'Stroids")
                .with_resizable(false)
                .with_inner_size(LogicalSize::new(1080, 720))
                .build(&event_loop)?;
        Ok (App {
            event_loop, window
        })
    }

    pub fn run(self, mut vulkan: VulkanInstance) {
        self.event_loop.run(move |event, _, control| {
            // Wait until next frame needs to be drawn
            let frame_time = Duration::from_secs_f32(1.0 / 60.0);
            *control = ControlFlow::WaitUntil(Instant::now() + frame_time);

            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } |
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some (VirtualKeyCode::Escape), ..
                        }, ..
                    }, ..
                } => {
                    *control = ControlFlow::Exit
                },
                Event::RedrawRequested(_) => {
                    let (image_index, _) = unsafe {
                        vulkan.swapchain_loader.acquire_next_image(vulkan.swapchain, u64::MAX, vulkan.semaphore_image_available[vulkan.swapchain_ptr], Fence::null()).expect("Failed to acquire image") // TODO
                    };

                    // Wait till the GPU is done with the image
                    unsafe {
                        vulkan.device.wait_for_fences(&[
                            vulkan.fence_draw_ready[vulkan.swapchain_ptr]
                        ], true, u64::MAX).expect("Waiting for fences");
                        vulkan.device.reset_fences(&[
                            vulkan.fence_draw_ready[vulkan.swapchain_ptr]
                        ]).expect("Resetting fences");
                    }

                    // Submit command buffer
                    let semaphores_available = [vulkan.semaphore_image_available[vulkan.swapchain_ptr]];
                    let waiting_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let semaphores_finished = [vulkan.semaphore_rendering_finished[vulkan.swapchain_ptr]];
                    let command_buffers = [vulkan.graphics_command_buffers[image_index as usize]];
                    let submit_info = [*SubmitInfo::builder()
                        .wait_semaphores(&semaphores_available)
                        .wait_dst_stage_mask(&waiting_stages)
                        .command_buffers(&command_buffers)
                        .signal_semaphores(&semaphores_finished)];
                    unsafe {
                        vulkan.device.queue_submit(
                            vulkan.graphics_queue,
                            &submit_info,
                            vulkan.fence_draw_ready[vulkan.swapchain_ptr]
                        ).expect("Submitting queue")
                    };

                    // Present next image on screen
                    let swapchains = [vulkan.swapchain];
                    let indices = [image_index];
                    let create_info = PresentInfoKHR::builder()
                        .wait_semaphores(&semaphores_finished)
                        .swapchains(&swapchains)
                        .image_indices(&indices);
                    unsafe {
                        vulkan.swapchain_loader.queue_present(
                            vulkan.graphics_queue,
                            &create_info)
                            .expect("Presenting next image from swapchain");
                    }

                    vulkan.swapchain_ptr = (vulkan.swapchain_ptr + 1) % vulkan.swapchain_image_views.len() as usize;
                },
                Event::MainEventsCleared => {
                    self.window.request_redraw();
                },
                _ => {}
            }
        })
    }
}
