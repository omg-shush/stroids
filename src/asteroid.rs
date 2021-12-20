use crate::region::Region;

pub enum AsteroidType {
    Silicate,
    Icy,
    Metallic,
    Carbon
}

pub struct Asteroid {
    asteroid_type: AsteroidType,
    size: [u32; 3],
    region: Region,
}

impl Asteroid {
    pub fn new(asteroid_type: AsteroidType, size: [u32; 3]) -> Asteroid {
        Asteroid { asteroid_type, size, region: Region::new(size) }
    }
}
