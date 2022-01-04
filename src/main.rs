use std::error::Error;

use asteroid::{Asteroid, AsteroidType};
use vulkan_instance::VulkanInstance;
use app::App;

mod asteroid;
mod region;
mod production;
mod vulkan_instance;
mod app;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");

    let app = App::new()?;
    let vinstance = VulkanInstance::new(&app.window)?;

    let asteroid = Asteroid::new(AsteroidType::Metallic, [100, 100, 30]);

    app.run();

    Ok(())
}
