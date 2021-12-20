#[derive(Clone)]
enum Voxel {
    Empty,
    Regolith,
    Silicate,
    Clay,
    Metallic,
    Ice,
    Carbonaceous
}

pub struct Region {
    voxels: Vec<Vec<Vec<Voxel>>> // [x][y][z]
}

impl Region {
    pub fn new(size: [u32; 3]) -> Region {
        let mut r = Region { voxels: vec![vec![vec![Voxel::Empty; size[2] as usize]; size[1] as usize]; size[0] as usize] };
        r.generate();
        r
    }

    fn generate(&mut self) {
        // First generate basic rock structures
    }
}
