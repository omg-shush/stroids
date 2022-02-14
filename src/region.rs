use ash::vk::CommandBuffer;
use nalgebra::Matrix4;

use crate::vulkan::vulkan_instance::VulkanInstance;

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
    width: u32, // x
    height: u32, // y
    depth: u32, // z
    voxels: Vec<Vec<Vec<Voxel>>> // [x][y][z]
}

impl Region {
    pub fn new(size: [u32; 3]) -> Region {
        let mut r = Region {
            width: size[0],
            height: size[1],
            depth: size[2],
            voxels: vec![vec![vec![Voxel::Empty; size[2] as usize]; size[1] as usize]; size[0] as usize]
        };
        r.generate();
        r
    }

    fn generate(&mut self) {
        // First generate basic rock structures
        self.voxels[0][0][0] = Voxel::Regolith;
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, view: Matrix4<f32>, view_rot: Matrix4<f32>, projection: Matrix4<f32>, time: f32) {
        
    }
}
