use std::error::Error;
use std::fmt::{Debug};
use std::slice;

use ash::Device;
use ash::vk::{Pipeline, PipelineLayout, PipelineLayoutCreateInfo, DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags, DescriptorSetLayoutCreateInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSetAllocateInfo, DescriptorSet, BufferUsageFlags, WriteDescriptorSet, DescriptorBufferInfo, WHOLE_SIZE, CommandPoolCreateInfo, CommandPoolCreateFlags, CommandPool, CommandBufferAllocateInfo, CommandBufferLevel, CommandPoolResetFlags, PipelineBindPoint, CommandBufferBeginInfo, CommandBufferUsageFlags, SubmitInfo, Fence, FenceCreateInfo, MemoryMapFlags, DescriptorPool, DescriptorSetLayout, Buffer, DescriptorImageInfo};
use nalgebra::{Vector3, vector};

use crate::buffer::DynamicBuffer;
use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;
use crate::vulkan::vulkan_pipeline::VulkanPipeline;

// Index corners of a cube as follows:
// 0b000 = min x, y, and z coords
// 0b100 = max x coord
// 0b010 = max y coord
// 0b001 = max z coord
#[derive(Clone, Copy)]
pub struct Cube {
    pattern: u8, // Bit vector of cube corners that are "inside" mesh
    edges: [(u8, u8); 15], // Edges on which this cube outputs a vertex, defined by pairs of cube corners
    edges_len: u8
}

impl Debug for Cube {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in 0..self.edges_len as usize {
            f.write_fmt(format_args!("({}, {}) ", self.edges[e].0, self.edges[e].1))?;
        }
        Ok (())
    }
}

impl Cube {
    fn new(pattern: &[u8], edges: &[(u8, u8)]) -> Cube {
        let mut p = 0u8;
        for corner in pattern {
            p |= 1 << corner;
        }
        let mut es = [(0, 0); 15];
        assert!(edges.len() % 3 == 0);
        es[..edges.len()].copy_from_slice(edges);
        Cube { pattern: p, edges: es, edges_len: edges.len() as u8 }
    }

    fn invert(&self) -> Cube {
        let pattern = !self.pattern;
        let edges = self.edges
            .chunks(3)
            .map(|tri| [tri[0], tri[2], tri[1]])
            .flatten()
            .collect::<Vec<_>>();
        let mut es = [(0, 0); 15];
        es[..self.edges_len as usize].copy_from_slice(&edges[..self.edges_len as usize]);

        Cube { pattern, edges: es, edges_len: self.edges_len }
    }

    fn reflect(&self, axis: u8) -> Cube {
        // For every bit in pattern, reflect it in the direction of axis
        let mut pattern = self.pattern;
        if axis & CUBE_X > 0 {
            // For x axis: map 76543210 to 32107654
            pattern = ((pattern & 0b1111_0000) >> 4) | ((pattern & 0b0000_1111) << 4);
        }
        if axis & CUBE_Y > 0 {
            // For y axis: map 76543210 to 54761032
            pattern = ((pattern & 0b1100_1100) >> 2) | ((pattern & 0b0011_0011) << 2);
        }
        if axis & CUBE_Z > 0 {
            // For z axis: map 76543210 to 67452301
            pattern = ((pattern & 0b1010_1010) >> 1) | ((pattern & 0b0101_0101) << 1);
        }

        if pattern == self.pattern {
            return self.clone();
        }

        let edges = self.edges
            .map(|(v1, v2)| (v1 ^ axis, v2 ^ axis))
            .chunks(3)
            .map(|tri| if axis.count_ones() % 2 == 0 { tri.try_into().unwrap() } else { [tri[0], tri[2], tri[1]] })
            .flatten()
            .collect::<Vec<_>>();
        let mut es = [(0, 0); 15];
        es[..self.edges_len as usize].copy_from_slice(&edges[..self.edges_len as usize]);
        Cube { pattern, edges: es, edges_len: self.edges_len }
    }

    fn rotate(&self, axis: u8) -> Cube {
        let mut pattern = self.pattern;
        let mut edges = self.edges;

        if axis & CUBE_Z > 0 {
            // Rotate 90 degrees clockwise about the Z axis
            // CUBE -> CUBE_Y -> CUBE_XY -> CUBE_X
            // Map 76543210 to 54107632
            // 76 shr 4, 54 shl 2, 32 shr 2, 10 shl 4
            pattern = ((pattern & 0xc0) >> 4) | ((pattern & 0x30) << 2)
                    | ((pattern & 0x0c) >> 2) | ((pattern & 0x03) << 4);
            pattern = ((pattern & 0xc0) >> 4) | ((pattern & 0x30) << 2)
                    | ((pattern & 0x0c) >> 2) | ((pattern & 0x03) << 4);
            pattern = ((pattern & 0xc0) >> 4) | ((pattern & 0x30) << 2)
                    | ((pattern & 0x0c) >> 2) | ((pattern & 0x03) << 4);
            fn rot_z(corner: u8) -> u8 {
                match corner {
                    CUBE => CUBE_Y,
                    CUBE_Y => CUBE_XY,
                    CUBE_XY => CUBE_X,
                    CUBE_X => CUBE,
                    CUBE_Z => CUBE_YZ,
                    CUBE_YZ => CUBE_XYZ,
                    CUBE_XYZ => CUBE_XZ,
                    CUBE_XZ => CUBE_Z,
                    _ => panic!("Illegal cube code")
                }
            }
            edges = edges.map(|(v1, v2)| (rot_z(v1), rot_z(v2)));
        }
        if axis & CUBE_Y > 0 {
            // Rotate 90 degrees clockwise about the Y axis
            // CUBE -> CUBE_X -> CUBE_XZ -> CUBE_Z
            // Map 76543210 to 37152604
            // 7_5 >> 1, 6_4 >> 4, 3_1 << 4, 2_0 << 1
            pattern = ((pattern & 0xa0) >> 1) | ((pattern & 0x50) >> 4)
                    | ((pattern & 0x0a) << 4) | ((pattern & 0x05) << 1);
            pattern = ((pattern & 0xa0) >> 1) | ((pattern & 0x50) >> 4)
                    | ((pattern & 0x0a) << 4) | ((pattern & 0x05) << 1);
            pattern = ((pattern & 0xa0) >> 1) | ((pattern & 0x50) >> 4)
                    | ((pattern & 0x0a) << 4) | ((pattern & 0x05) << 1);
            fn rot_y(corner: u8) -> u8 {
                match corner {
                    CUBE => CUBE_X,
                    CUBE_X => CUBE_XZ,
                    CUBE_XZ => CUBE_Z,
                    CUBE_Z => CUBE,
                    CUBE_Y => CUBE_XY,
                    CUBE_XY => CUBE_XYZ,
                    CUBE_XYZ => CUBE_YZ,
                    CUBE_YZ => CUBE_Y,
                    _ => panic!("Illegal cube code")
                }
            }
            edges = edges.map(|(v1, v2)| (rot_y(v1), rot_y(v2)));
        }
        if axis & CUBE_X > 0 {
            // Rotate 90 degrees clockwise about the X axis
            // CUBE -> CUBE_Z -> CUBE_YZ -> CUBE_Y
            // Map 76543210 to 64752031
            // 7___3 >> 2, 6___2 << 1, 5___1 >> 1, 4___0 << 2
            pattern = ((pattern & 0x88) >> 2) | ((pattern & 0x44) << 1)
                    | ((pattern & 0x22) >> 1) | ((pattern & 0x11) << 2);
            pattern = ((pattern & 0x88) >> 2) | ((pattern & 0x44) << 1)
                    | ((pattern & 0x22) >> 1) | ((pattern & 0x11) << 2);
            pattern = ((pattern & 0x88) >> 2) | ((pattern & 0x44) << 1)
                    | ((pattern & 0x22) >> 1) | ((pattern & 0x11) << 2);
            fn rot_x(corner: u8) -> u8 {
                match corner {
                    CUBE => CUBE_Z,
                    CUBE_Z => CUBE_YZ,
                    CUBE_YZ => CUBE_Y,
                    CUBE_Y => CUBE,
                    CUBE_X => CUBE_XZ,
                    CUBE_XZ => CUBE_XYZ,
                    CUBE_XYZ => CUBE_XY,
                    CUBE_XY => CUBE_X,
                    _ => panic!("Illegal cube code")
                }
            }
            edges = edges.map(|(v1, v2)| (rot_x(v1), rot_x(v2)));
        }

        let mut es = [(0, 0); 15];
        es[..edges.len()].copy_from_slice(&edges);
        Cube { pattern, edges: es, edges_len: self.edges_len }
    }
}

pub const CUBE:     u8 = 0b000;
pub const CUBE_X:   u8 = 0b100;
pub const CUBE_Y:   u8 = 0b010;
pub const CUBE_Z:   u8 = 0b001;
pub const CUBE_XY:  u8 = 0b110;
pub const CUBE_XZ:  u8 = 0b101;
pub const CUBE_YZ:  u8 = 0b011;
pub const CUBE_XYZ: u8 = 0b111;

pub struct MarchingCubes {
    device: Device,
    triangulation: DynamicBuffer,
    compute: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set_layout: DescriptorSetLayout,
    output_descriptor: DescriptorSet,
    fence: Fence
}

impl Drop for MarchingCubes {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.compute, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl MarchingCubes {
    pub fn new(vulkan: &VulkanInstance) -> Result<MarchingCubes, Box<dyn Error>> {
        let basis = [ // 0, 1, 2
            Cube::new(&[], &[]),
            Cube::new(&[CUBE], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z)]),
            Cube::new(&[CUBE, CUBE_X], &[(CUBE, CUBE_Z), (CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Y), (CUBE, CUBE_Z)]),
            // 3, 4, 5
            Cube::new(&[CUBE, CUBE_XY], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_XY, CUBE_XYZ), (CUBE_Y, CUBE_XY)]),
            Cube::new(&[CUBE, CUBE_XYZ], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ)]),
            Cube::new(&[CUBE_X, CUBE_XZ, CUBE_Z], &[(CUBE, CUBE_X), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Z), (CUBE_Z, CUBE_YZ), (CUBE_Z, CUBE_YZ), (CUBE_XZ, CUBE_XYZ), (CUBE_X, CUBE_XY)]),
            // 6, 7, 8
            Cube::new(&[CUBE, CUBE_X, CUBE_XYZ], &[(CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ)]),
            Cube::new(&[CUBE_X, CUBE_Y, CUBE_XYZ], &[(CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_YZ), (CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ)]),
            Cube::new(&[CUBE, CUBE_X, CUBE_Z, CUBE_XZ], &[(CUBE_X, CUBE_XY), (CUBE, CUBE_Y), (CUBE_XZ, CUBE_XYZ), (CUBE_XZ, CUBE_XYZ), (CUBE, CUBE_Y), (CUBE_Z, CUBE_YZ)]),
            // 9, 10, 11
            Cube::new(&[CUBE, CUBE_Z, CUBE_XZ, CUBE_YZ], &[(CUBE_X, CUBE_XZ), (CUBE, CUBE_X), (CUBE_XZ, CUBE_XYZ), (CUBE_XZ, CUBE_XYZ), (CUBE, CUBE_X), (CUBE_YZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE, CUBE_Y), (CUBE_Y, CUBE_YZ), (CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE_YZ, CUBE_XYZ)]),
            Cube::new(&[CUBE, CUBE_Y, CUBE_XZ, CUBE_XYZ], &[(CUBE, CUBE_X), (CUBE_Y, CUBE_XY), (CUBE, CUBE_Z), (CUBE, CUBE_Z), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_YZ), (CUBE_X, CUBE_XZ), (CUBE_Z, CUBE_XZ), (CUBE_XY, CUBE_XYZ), (CUBE_XY, CUBE_XYZ), (CUBE_Z, CUBE_XZ), (CUBE_YZ, CUBE_XYZ)]),
            Cube::new(&[CUBE, CUBE_Z, CUBE_XZ, CUBE_XYZ], &[(CUBE_X, CUBE_XZ), (CUBE, CUBE_X), (CUBE_XY, CUBE_XYZ), (CUBE_XY, CUBE_XYZ), (CUBE, CUBE_X), (CUBE_Z, CUBE_YZ), (CUBE_Z, CUBE_YZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ), (CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE_Z, CUBE_YZ)]),
            // 12, 13, 14
            Cube::new(&[CUBE_X, CUBE_Y, CUBE_Z, CUBE_XZ], &[(CUBE, CUBE_Y), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_YZ), (CUBE, CUBE_X), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Z), (CUBE_Z, CUBE_YZ), (CUBE_Z, CUBE_YZ), (CUBE_XZ, CUBE_XYZ), (CUBE_X, CUBE_XY)]),
            Cube::new(&[CUBE, CUBE_XZ, CUBE_XY, CUBE_YZ], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_Z, CUBE_YZ), (CUBE_Y, CUBE_YZ), (CUBE_YZ, CUBE_XYZ), (CUBE_X, CUBE_XZ), (CUBE_Z, CUBE_XZ), (CUBE_XZ, CUBE_XYZ), (CUBE_X, CUBE_XY), (CUBE_XY, CUBE_XYZ), (CUBE_Y, CUBE_XY)]),
            Cube::new(&[CUBE_X, CUBE_Z, CUBE_XZ, CUBE_YZ], &[(CUBE, CUBE_X), (CUBE, CUBE_Z), (CUBE_Y, CUBE_YZ), (CUBE, CUBE_X), (CUBE_Y, CUBE_YZ), (CUBE_XZ, CUBE_XYZ), (CUBE, CUBE_X), (CUBE_XZ, CUBE_XYZ), (CUBE_X, CUBE_XY), (CUBE_XZ, CUBE_XYZ), (CUBE_Y, CUBE_YZ), (CUBE_YZ, CUBE_XYZ)]),
            // 15, 16, 17
            Cube::new(&[CUBE, CUBE_X, CUBE_Y, CUBE_Z, CUBE_XYZ], &[(CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ), (CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_XY), (CUBE_Z, CUBE_XZ), (CUBE_X, CUBE_XZ), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_YZ), (CUBE_Z, CUBE_XZ), (CUBE_Z, CUBE_XZ), (CUBE_Y, CUBE_YZ), (CUBE_Z, CUBE_YZ)]),
            Cube::new(&[CUBE, CUBE_X, CUBE_Y, CUBE_YZ, CUBE_XYZ], &[(CUBE_XY, CUBE_XYZ), (CUBE_XZ, CUBE_XYZ), (CUBE_Y, CUBE_XY), (CUBE_Y, CUBE_XY), (CUBE_XZ, CUBE_XYZ), (CUBE_Z, CUBE_YZ), (CUBE_Y, CUBE_XY), (CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE_Y, CUBE_XY), (CUBE, CUBE_Z), (CUBE_X, CUBE_XZ), (CUBE_Y, CUBE_XY), (CUBE_Z, CUBE_YZ), (CUBE, CUBE_Z)]),
            Cube::new(&[CUBE, CUBE_X, CUBE_Y, CUBE_XZ, CUBE_YZ, CUBE_XY], &[(CUBE, CUBE_Z), (CUBE_Z, CUBE_XZ), (CUBE_XZ, CUBE_XYZ), (CUBE, CUBE_Z), (CUBE_XZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ), (CUBE, CUBE_Z), (CUBE_XY, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE, CUBE_Z), (CUBE_YZ, CUBE_XYZ), (CUBE_Z, CUBE_YZ)])
        ];

        let rotated: [Cube; 72] = [
                basis,
                basis.map(|c| c.rotate(CUBE_X)),
                basis.map(|c| c.rotate(CUBE_Y)),
                basis.map(|c| c.rotate(CUBE_Z))
            ].concat().as_slice().try_into().unwrap();
        let reflected = [
                rotated,
                rotated.map(|c| c.reflect(CUBE_X)),
                rotated.map(|c| c.reflect(CUBE_Y)),
                rotated.map(|c| c.reflect(CUBE_Z)),
                rotated.map(|c| c.reflect(CUBE_XY)),
                rotated.map(|c| c.reflect(CUBE_XZ)),
                rotated.map(|c| c.reflect(CUBE_YZ)),
                rotated.map(|c| c.reflect(CUBE_XYZ))
            ].concat();
        let inverted = reflected.iter().map(|c| c.invert());
        let mut cubes = Vec::new();
        cubes.extend(&reflected);
        cubes.extend(inverted);
        cubes.sort_by_key(|c| c.pattern);
        cubes.dedup_by_key(|c| c.pattern);

        // Write triangulation array to a buffer for use in compute shader
        let triangulation_array = cubes.iter().flat_map(|cube| {
            let mut corner_pairs = cube.edges.to_vec();
            corner_pairs.truncate(cube.edges_len as usize);
            let mut vertices = corner_pairs.into_iter().map(|(a, b)| {
                let ax = if a & CUBE_X > 0 { 1 } else { 0 };
                let ay = if a & CUBE_Y > 0 { 1 } else { 0 };
                let az = if a & CUBE_Z > 0 { 1 } else { 0 };
                let bx = if b & CUBE_X > 0 { 1 } else { 0 };
                let by = if b & CUBE_Y > 0 { 1 } else { 0 };
                let bz = if b & CUBE_Z > 0 { 1 } else { 0 };
                let nx = (ax + bx) as f32 / 2.0;
                let ny = (ay + by) as f32 / 2.0;
                let nz = (az + bz) as f32 / 2.0;
                vector![nx, ny, nz]
            }).collect::<Vec<_>>();
            vertices.extend([vector![f32::NAN, f32::NAN, f32::NAN]].iter().cycle().take(15 - vertices.len()));
            vertices.into_iter().flat_map(|v| [v[0], v[1], v[2]])
        }).collect::<Vec<_>>();
        let triangulation = DynamicBuffer::new(&vulkan, &triangulation_array, BufferUsageFlags::STORAGE_BUFFER)?;

        // Init compute pipeline
        let (pipeline, pipeline_layout, descriptor_set_layout) = {
            let descriptor_set_layout = {
                let descriptor_bindings = [
                    DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(ShaderStageFlags::COMPUTE)
                        .build(),
                    DescriptorSetLayoutBinding::builder()
                        .binding(1)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(ShaderStageFlags::COMPUTE)
                        .build()
                ];
                let create_info = DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&descriptor_bindings);
                unsafe { vulkan.device.create_descriptor_set_layout(&create_info, None)? }
            };
            let noise_image_layout = {
                let noise_image_bindings = [
                    DescriptorSetLayoutBinding::builder()
                        .binding(0)
                        .descriptor_type(DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(ShaderStageFlags::COMPUTE)
                        .build()
                ];
                let create_info = DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&noise_image_bindings);
                unsafe { vulkan.device.create_descriptor_set_layout(&create_info, None)? }
            };
            let push_constant = ash::vk::PushConstantRange {
                stage_flags: ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 4 * 6 // 6 uint/int's
            };
            let set_layouts = [descriptor_set_layout, noise_image_layout];
            let push_constant_ranges = [push_constant];
            let pipeline_layout = PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);
            let (pipeline, pipeline_layout) = VulkanPipeline::new_compute(&vulkan.device, *pipeline_layout)?;

            // Cleanup useless layouts
            unsafe { vulkan.device.destroy_descriptor_set_layout(noise_image_layout, None) };
            
            (pipeline, pipeline_layout, descriptor_set_layout)
        };

        // Init buffer descriptors for compute
        let descriptor_pool = {
            let create_info = DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[DescriptorPoolSize {
                    ty: DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 2
                }]);
            unsafe { vulkan.device.create_descriptor_pool(&create_info, None)? }
        };
        vulkan.descriptor_pools.borrow_mut().push(descriptor_pool); // Register for cleanup
        let output_descriptor = {
            let set_layouts = [descriptor_set_layout];
            let create_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe { vulkan.device.allocate_descriptor_sets(&create_info)?[0] }
        };

        // Init signalling fence
        let fence = unsafe {
            let create_info = FenceCreateInfo::builder();
            vulkan.device.create_fence(&create_info, None)?
        };

        Ok (MarchingCubes {
            device: vulkan.device.clone(),
            compute: pipeline,
            triangulation, pipeline_layout, descriptor_set_layout, output_descriptor, fence
        })
    }

    // Generates a list of vertices by applying the marching cubes algorithm to the given noise function over the given range
    pub fn march(&self, vulkan: &VulkanInstance, start: Vector3<i32>, end: Vector3<i32>, noise: &Texture) -> Result<Vec<Vector3<f32>>, Box<dyn Error>> {
        // Allocate storage buffer // TODO deallocate afterwards!
        // Maximum 5 triangles = 15 vec3's = 45 f32's = 180 bytes per cube
        let num_cubes = (end - start).abs().product();
        let (storage, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::STORAGE_BUFFER, 180 * num_cubes as u64)?;
        
        // Attach buffers to descriptors
        unsafe {
            vulkan.device.update_descriptor_sets(&[
                WriteDescriptorSet::builder()
                    .dst_set(self.output_descriptor)
                    .dst_binding(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[DescriptorBufferInfo {
                        buffer: self.triangulation.buffer,
                        offset: 0,
                        range: WHOLE_SIZE
                    }])
                    .build(),
                WriteDescriptorSet::builder()
                    .dst_set(self.output_descriptor)
                    .dst_binding(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[DescriptorBufferInfo {
                        buffer: storage,
                        offset: 0,
                        range: WHOLE_SIZE
                    }])
                    .build()
            ], &[])
        };
        
        // Dispatch compute
        let cmdbuf = vulkan.graphics_command_buffers[0];
        let local_size = vector![10, 10, 10];
        let total_size = end - start;
        let group_count = total_size.cast::<f32>().component_div(&local_size.cast::<f32>()).map(|f| f.ceil() as u32);
        let pc_values = [total_size.as_slice(), start.as_slice()].concat();
        let pc = unsafe { slice::from_raw_parts(pc_values.as_ptr() as *const u8, 6 * 4) };
        unsafe {
            let begin_info = CommandBufferBeginInfo::builder()
                .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vulkan.device.begin_command_buffer(cmdbuf, &begin_info)?;
            vulkan.device.cmd_bind_pipeline(cmdbuf, PipelineBindPoint::COMPUTE, self.compute);
            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::COMPUTE, self.pipeline_layout,
                0, &[self.output_descriptor, noise.descriptor_set], &[]);
            vulkan.device.cmd_push_constants(cmdbuf, self.pipeline_layout, ShaderStageFlags::COMPUTE, 0, &pc);
            vulkan.device.cmd_dispatch(cmdbuf, group_count[0], group_count[1], group_count[2]);
            vulkan.device.end_command_buffer(cmdbuf)?;
            vulkan.device.queue_submit(vulkan.graphics_queue, &[SubmitInfo::builder()
                .command_buffers(&[cmdbuf])
                .build()
            ], self.fence)?;
            println!("Generating terrain...");
            vulkan.device.wait_for_fences(&[self.fence], true, 60_000_000_000)?; // TODO 1 min timeout
        }

        // Recover results from buffer
        let vertices = unsafe {
            let buf = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            // Maximum 5 triangles = 15 vec3's = 45 f32's = 180 bytes per cube
            let floats = slice::from_raw_parts(buf as *mut f32, 45 * num_cubes as usize);
            let vertices = floats.chunks_exact(3).map(|vert| vector![vert[0], vert[1], vert[2]]).filter(|v| !v[0].is_nan()).collect::<Vec<_>>();
            vulkan.device.unmap_memory(allocation.memory);
            vertices
        };

        // Cleanup compute
        unsafe {
            vulkan.device.reset_fences(&[self.fence])?;
        }

        Ok (vertices)
    }
}
