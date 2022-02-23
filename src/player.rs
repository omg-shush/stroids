use std::collections::HashMap;
use std::f32::consts::PI;
use std::rc::Rc;
use std::slice;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use ash::vk::{CommandBuffer, PipelineBindPoint, ShaderStageFlags, IndexType, BufferUsageFlags};
use nalgebra::{UnitQuaternion, Translation3, Vector3, Scale3, Rotation3, UnitVector3};
use obj::{Obj, TexturedVertex, load_obj};
use winit::event::VirtualKeyCode;

use crate::buffer::DynamicBuffer;
use crate::physics::{PhysicsEngine, Entity, EntityProperties, Mesh};
use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;

pub struct Player {
    pub camera: UnitQuaternion<f32>,
    vertices: DynamicBuffer,
    indices: DynamicBuffer,
    texture: Texture,
    pub entity: Entity
}

impl Player {
    pub fn new(vulkan: &VulkanInstance, physics: &mut PhysicsEngine) -> Result<Player, Box<dyn Error>> {
        let file = BufReader::new(File::open("res/rover.obj")?);
        let rocket: Obj<TexturedVertex> = load_obj(file)?;

        let texture = Texture::new(vulkan, "res/rover.png")?;

        let camera = UnitQuaternion::identity();

        // Allocate & write vertex/index buffers
        let vertices = rocket.vertices.iter().map(|v| {
            let mut vec = [v.position, v.normal].concat();
            vec.extend_from_slice(&v.texture[..2]);
            vec
        }).flatten().collect::<Vec<_>>();
        let vertices = DynamicBuffer::new(vulkan, &vertices, BufferUsageFlags::VERTEX_BUFFER)?;
        let indices = DynamicBuffer::new(vulkan, &rocket.indices, BufferUsageFlags::INDEX_BUFFER)?;

        let entity = physics.add_entity(EntityProperties { immovable: false, collision: false, gravitational: false });
        physics.set_entity(entity).position = Vector3::from([0.05, 3.68, 0.05]);
        physics.set_entity(entity).scale = Vector3::from([0.02, 0.02, 0.02]);
        physics.set_entity(entity).mass = 1.0;
        let physics_vertices = Rc::new(vec![
            Vector3::from([-1.0, 0.0, -1.0]),
            Vector3::from([-1.0, 0.0, 1.0]),
            Vector3::from([1.0, 0.0, 1.0]),
            Vector3::from([1.0, 0.0, -1.0])]);
        let physics_indices = vec![0, 1, 2, 2, 3, 0];
        physics.set_entity(entity).vertices = physics_vertices.clone();
        physics.set_entity(entity).mesh.push(Mesh::new(physics_vertices, physics_indices));

        Ok (Player {
            camera,
            vertices, indices,
            texture,
            entity
        })
    }

    pub fn update(&mut self, physics: &mut PhysicsEngine, keys: &HashMap<VirtualKeyCode, bool>, delta_time: f32) {
        let entity = physics.set_entity(self.entity);

        let _rocket_x_axis = UnitVector3::new_normalize(entity.rotation.transform_vector(&Vector3::x()));
        let rocket_y_axis = UnitVector3::new_normalize(entity.rotation.transform_vector(&Vector3::y()));
        let rocket_z_axis = UnitVector3::new_normalize(entity.rotation.transform_vector(&Vector3::z()));

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
        entity.rotation = rotation * entity.rotation;
        self.camera = rotation * self.camera;

        if *keys.get(&VirtualKeyCode::W).unwrap_or(&false) {
            entity.rotation = UnitQuaternion::new_normalize(entity.rotation.lerp(&self.camera, 0.07));
            entity.velocity += rocket_z_axis.scale(-0.15 * delta_time);
        } else if *keys.get(&VirtualKeyCode::S).unwrap_or(&false) {
            if entity.velocity.norm() > 0.01 {
                let inverse_momentum = UnitQuaternion::look_at_rh(&(-1.0 * entity.velocity), &entity.velocity.cross(&entity.rotation.transform_vector(&Vector3::x_axis())));
                entity.rotation = entity.rotation.try_slerp(&inverse_momentum, 0.07, 0.01).unwrap_or_else(||
                    UnitQuaternion::new_normalize(entity.rotation.lerp(&inverse_momentum, 0.07)));
            }
            entity.velocity *= 0.6f32.powf(delta_time);
        }
    }

    pub fn render(&self, vulkan: &VulkanInstance, physics: &PhysicsEngine, cmdbuf: CommandBuffer, vp: &[f32]) {
        let entity = physics.get_entity(self.entity);
        unsafe {
            let data = 0u32.to_ne_bytes(); // Turn lighting off for skybox
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::FRAGMENT, 128, &data);

            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.vertices.buffer], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmdbuf, self.indices.buffer, 0, IndexType::UINT16);

            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.texture.descriptor_set], &[]);

            let model = Translation3::from(entity.position).to_homogeneous()
                * entity.rotation.to_homogeneous()
                * Rotation3::from_axis_angle(&Vector3::z_axis(), PI).to_homogeneous()
                // * Rotation3::rotation_between(&Vector3::x(), &Vector3::y()).unwrap().to_homogeneous() // TODO why option?
                * Scale3::from(entity.scale).to_homogeneous();
                // * Translation3::from([0.0, 0.21, 0.0]).to_homogeneous();
            let data = [model.as_slice(), vp].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_draw_indexed(cmdbuf, self.indices.len, 1, 0, 0, 0);
        }
    }
}
