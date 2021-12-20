use crate::asteroid::{Asteroid, AsteroidType};

mod asteroid;
mod region;
mod production;

fn main() {
    println!("Hello, world!");

    let asteroid = Asteroid::new(AsteroidType::Metallic, [100, 100, 30]);
}
