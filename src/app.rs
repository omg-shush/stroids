use std::error::Error;
use std::time::{Instant, Duration};

use winit::dpi::LogicalSize;
use winit::event::{WindowEvent, Event, KeyboardInput, ElementState, VirtualKeyCode};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};

pub struct App {
    event_loop: EventLoop<()>,
    window: Window
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

    pub fn run(self) {
        self.event_loop.run(|event, _, control| {
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
                Event::MainEventsCleared => {
                    // TODO render
                },
                _ => {}
            }
        })
    }
}
