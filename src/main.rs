use std::error::Error;

use asteroid::{Asteroid, AsteroidType};
use vulkan::vulkan_instance::VulkanInstance;
use app::App;
use system::System;

mod asteroid;
mod region;
mod production;
mod vulkan;
mod app;
mod system;
mod texture;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");

    let app = App::new()?;
    let vulkan = VulkanInstance::new(&app.window)?;

    let asteroid = Asteroid::new(AsteroidType::Metallic, [100, 100, 30]);
    let system = System::new(&vulkan)?;

    app.run(vulkan, system);

    Ok(())
}
