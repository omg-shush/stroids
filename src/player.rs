use std::collections::HashMap;
use std::slice;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use ash::vk::{Buffer, BufferUsageFlags, MemoryMapFlags, CommandBuffer, PipelineBindPoint, ShaderStageFlags, IndexType};
use nalgebra::{UnitQuaternion, Translation3, Vector3, Scale3, Rotation3, UnitVector3};
use obj::{Obj, TexturedVertex, load_obj};
use winit::event::VirtualKeyCode;

use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;

pub struct Player {
    pub camera: UnitQuaternion<f32>,
    pub orientation: UnitQuaternion<f32>,
    pub velocity: Vector3<f32>,
    pub position: Vector3<f32>,
    // TODO create "model" object
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
    //
    texture: Texture
}

impl Player {
    pub fn new(vulkan: &VulkanInstance) -> Result<Player, Box<dyn Error>> {
        let file = BufReader::new(File::open("res/rocket.obj")?);
        let rocket: Obj<TexturedVertex> = load_obj(file)?;

        let texture = Texture::new(vulkan, "res/rocket.png")?;

        let camera = UnitQuaternion::identity();
        let orientation = UnitQuaternion::identity();
        let velocity = Vector3::zeros();

        // Allocate & write vertex buffer
        let vertices = rocket.vertices.iter().map(|v| {
            let mut vec = [v.position, v.normal].concat();
            vec.extend_from_slice(&v.texture[..2]);
            vec
        }).flatten().collect::<Vec<_>>();
        let (vertex_buffer, _vertex_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::VERTEX_BUFFER, (vertices.len() * 4) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut f32).copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        // Allocate & write index buffer
        let (index_buffer, _index_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::INDEX_BUFFER, (rocket.indices.len() * 2) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut u16).copy_from_nonoverlapping(rocket.indices.as_ptr(), rocket.indices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        Ok (Player {
            camera, orientation, velocity, position: Vector3::from([5.0, 0.0, 0.0]),
            vertex_buffer, index_buffer, index_count: rocket.indices.len() as u32,
            texture
        })
    }

    pub fn update(&mut self, keys: &HashMap<VirtualKeyCode, bool>, delta_time: f32) {
        let _rocket_x_axis = UnitVector3::new_normalize(self.orientation.transform_vector(&Vector3::x()));
        let rocket_y_axis = UnitVector3::new_normalize(self.orientation.transform_vector(&Vector3::y()));
        let rocket_z_axis = UnitVector3::new_normalize(self.orientation.transform_vector(&Vector3::z()));

        self.position += self.velocity * delta_time;

        let rotation =
            if *keys.get(&VirtualKeyCode::A).unwrap_or(&false) {
                UnitQuaternion::from_axis_angle(&rocket_y_axis, 0.2 * delta_time)
            } else if *keys.get(&VirtualKeyCode::D).unwrap_or(&false) {
                UnitQuaternion::from_axis_angle(&rocket_y_axis, -0.2 * delta_time)
            } else if *keys.get(&VirtualKeyCode::Q).unwrap_or(&false) {
                UnitQuaternion::from_axis_angle(&rocket_z_axis, -0.2 * delta_time)
            } else if *keys.get(&VirtualKeyCode::E).unwrap_or(&false) {
                UnitQuaternion::from_axis_angle(&rocket_z_axis, 0.2 * delta_time)
            } else {
                UnitQuaternion::identity()
            };
        self.orientation = rotation * self.orientation;
        self.camera = rotation * self.camera;

        if *keys.get(&VirtualKeyCode::W).unwrap_or(&false) {
            self.orientation = UnitQuaternion::new_normalize(self.orientation.lerp(&self.camera, 0.07));
            self.velocity += rocket_z_axis.scale(-0.4 * delta_time);
        } else if *keys.get(&VirtualKeyCode::S).unwrap_or(&false) {
            if self.velocity.norm() > 0.001 {
                let inverse_momentum = UnitQuaternion::look_at_rh(&(-1.0 * self.velocity), &self.velocity.cross(&self.orientation.transform_vector(&Vector3::x_axis())));
                self.orientation = self.orientation.try_slerp(&inverse_momentum, 0.07, 0.01).unwrap_or_else(||
                    UnitQuaternion::new_normalize(self.orientation.lerp(&inverse_momentum, 0.07)));
                }
            self.velocity *= 0.6f32.powf(delta_time);
        }
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, vp: &[f32]) {
        unsafe {
            let data = 0u32.to_ne_bytes(); // Turn lighting off for skybox
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::FRAGMENT, 128, &data);

            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertex_buffer], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmdbuf, self.index_buffer, 0, IndexType::UINT16);

            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.texture.descriptor_set], &[]);

            let model = Translation3::from(self.position).to_homogeneous()
                * self.orientation.to_homogeneous()
                * Rotation3::rotation_between(&Vector3::z(), &Vector3::y()).unwrap().to_homogeneous() // TODO why option?
                * Scale3::from([0.02, 0.02, 0.02]).to_homogeneous();
            let data = [model.as_slice(), vp].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_draw_indexed(cmdbuf, self.index_count, 1, 0, 0, 0);
        }
    }
}
