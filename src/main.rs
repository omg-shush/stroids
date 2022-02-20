use std::error::Error;

use asteroid::{Asteroid, AsteroidType};
use vulkan::vulkan_instance::VulkanInstance;
use app::App;
use system::System;
use physics::PhysicsEngine;
use player::Player;

mod asteroid;
mod region;
mod production;
mod vulkan;
mod app;
mod system;
mod texture;
mod player;
mod buffer;
mod physics;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");

    let app = App::new()?;
    let vulkan = VulkanInstance::new(&app.window)?;

    let mut physics = PhysicsEngine::new();
    let player = Player::new(&vulkan, &mut physics)?;
    let asteroid = Asteroid::new(&vulkan, &mut physics, AsteroidType::Metallic, [100, 100, 30])?;
    let system = System::new(&vulkan)?;

    app.run(vulkan, asteroid, system, physics, player);

    Ok(())
}
