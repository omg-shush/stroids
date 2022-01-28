use std::error::Error;
use std::time::{Instant, Duration};

use winit::dpi::LogicalSize;
use winit::event::{WindowEvent, Event, KeyboardInput, ElementState, VirtualKeyCode};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use ash::vk::{PipelineStageFlags, SubmitInfo};

use crate::system::System;
use crate::vulkan::vulkan_instance::VulkanInstance;

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

    pub fn run(self, mut vulkan: VulkanInstance, system: System) {
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
                    *control = ControlFlow::Exit;
                },
                Event::RedrawRequested(_) => {
                    vulkan.swapchain.acquire(&vulkan.device);

                    // Rewrite command buffer
                    let cmdbuf = vulkan.begin_commands().expect("Failed to begin recording commands");
                    system.render(&vulkan, cmdbuf);
                    unsafe {
                        vulkan.device.cmd_end_render_pass(cmdbuf);
                        vulkan.device.end_command_buffer(cmdbuf).expect("Failed to end recording commands");
                    };

                    // Submit command buffer
                    let semaphores_available = [vulkan.swapchain.semaphore_image_available[vulkan.swapchain.index]];
                    let waiting_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let semaphores_finished = [vulkan.swapchain.semaphore_rendering_finished[vulkan.swapchain.index]];
                    let command_buffers = [vulkan.graphics_command_buffers[vulkan.swapchain.index]];
                    let submit_info = [*SubmitInfo::builder()
                        .wait_semaphores(&semaphores_available)
                        .wait_dst_stage_mask(&waiting_stages)
                        .command_buffers(&command_buffers)
                        .signal_semaphores(&semaphores_finished)];
                    unsafe {
                        vulkan.device.queue_submit(
                            vulkan.graphics_queue,
                            &submit_info,
                            vulkan.swapchain.fence_resources_ready[vulkan.swapchain.index]
                        ).expect("Submitting queue")
                    };

                    vulkan.swapchain.present(&vulkan.graphics_queue);
                },
                Event::MainEventsCleared => {
                    self.window.request_redraw();
                },
                _ => {}
            }
        })
    }
}
