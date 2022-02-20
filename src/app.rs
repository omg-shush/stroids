use std::collections::HashMap;
use std::error::Error;
use std::time::{Instant, Duration};

use nalgebra::{Vector3, Translation3, Perspective3, UnitQuaternion, UnitVector3};
use winit::dpi::LogicalSize;
use winit::event::{WindowEvent, Event, KeyboardInput, ElementState, VirtualKeyCode, DeviceEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use ash::vk::{PipelineStageFlags, SubmitInfo};

use crate::asteroid::Asteroid;
use crate::physics::PhysicsEngine;
use crate::player::Player;
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

    pub fn run(self, mut vulkan: VulkanInstance, asteroid: Asteroid, system: System, mut physics: PhysicsEngine, mut player: Player) {
        self.window.set_cursor_grab(true).expect("Failed to grab cursor");
        self.window.set_cursor_visible(false);

        let time_start = Instant::now();
        let mut last_frame = time_start;
        let mut last_second = time_start;
        let mut fps = 0;
        let mut keys: HashMap<VirtualKeyCode, bool> = HashMap::new();

        self.event_loop.run(move |event, _, control| {
            // Wait until next frame needs to be drawn
            let frame_time = Duration::from_secs_f32(1.0 / 60.0);
            *control = ControlFlow::WaitUntil(Instant::now() + frame_time);

            let camera_x_axis = UnitVector3::new_normalize(player.camera.transform_vector(&Vector3::x()));
            let camera_y_axis = UnitVector3::new_normalize(player.camera.transform_vector(&Vector3::y()));
            let _camera_z_axis = UnitVector3::new_normalize(player.camera.transform_vector(&Vector3::z()));

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
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state,
                            virtual_keycode: Some (key), ..
                        }, ..
                    }, ..
                } => {
                    keys.insert(key, state == ElementState::Pressed);
                },
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion {
                        delta: (x, y)
                    }, ..
                } => {
                    player.camera =
                        UnitQuaternion::from_axis_angle(&camera_y_axis, -0.001 * x as f32)
                        * UnitQuaternion::from_axis_angle(&camera_x_axis, 0.001 * y as f32)
                        * player.camera;
                },
                Event::RedrawRequested(_) => {
                    vulkan.swapchain.acquire(&vulkan.device);
                    let now = Instant::now();
                    let time = (now - time_start).as_secs_f32();
                    let delta_time = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    // FPS counter
                    fps += 1;
                    if now - last_second > Duration::from_secs(1) {
                        last_second = now;
                        println!("FPS: {}", fps);
                        fps = 0;
                    }

                    // Update
                    player.update(&mut physics, &keys, delta_time);
                    physics.time_step(delta_time);

                    // Rewrite command buffer
                    let cmdbuf = vulkan.begin_commands().expect("Failed to begin recording commands");

                    let (view, view_rot) = {
                        let pos = physics.get_entity(player.entity).position + player.camera.transform_vector(&Vector3::from([0.0, -0.08, 0.5]));
                        let t = Translation3::from(-1.0 * pos);
                        let r = player.camera.inverse().to_rotation_matrix();
                        (r.to_homogeneous() * t.to_homogeneous(), r.to_homogeneous())
                    };
                    let projection = {
                        Perspective3::new(1080.0 / 720.0, 70f32.to_radians(), 0.001, 200.0).to_homogeneous()
                    };

                    // Render
                    system.render(&vulkan, cmdbuf, view, view_rot, projection, time);
                    asteroid.render(&vulkan, &physics, cmdbuf, projection * view);
                    player.render(&vulkan, &physics, cmdbuf, (projection * view).as_slice());

                    unsafe {
                        vulkan.device.cmd_end_render_pass(cmdbuf);
                        vulkan.device.end_command_buffer(cmdbuf).expect("Failed to end recording commands");
                    };

                    // Submit command buffer
                    let semaphores_available = [vulkan.swapchain.semaphore_image_available[vulkan.swapchain.index]];
                    let waiting_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let semaphores_finished = [vulkan.swapchain.semaphore_rendering_finished[vulkan.swapchain.index]];
                    let command_buffers = [cmdbuf];
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
