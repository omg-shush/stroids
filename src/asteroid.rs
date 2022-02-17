use std::collections::hash_map;
use std::error::Error;
use std::f32::consts::{PI, FRAC_PI_2};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::slice;
use std::cmp::Ordering::Equal;
use std::convert::From;

use ash::vk::{CommandBuffer, ShaderStageFlags, PipelineBindPoint, IndexType};
use nalgebra::{Matrix4, Vector3, Translation3, Scale3, Rotation3};
use obj::{Obj, TexturedVertex, load_obj};
use rand::{thread_rng, Rng};

use crate::buffer::DynamicBuffer;
use crate::region::Region;
use crate::texture::Texture;
use crate::vulkan::vulkan_instance::VulkanInstance;

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
    terrain: DynamicBuffer,
    indices: DynamicBuffer,
    texture: Texture
}

impl Asteroid {
    pub fn new(vulkan: &VulkanInstance, asteroid_type: AsteroidType, size: [u32; 3]) -> Result<Asteroid, Box<dyn Error>> {
        let perlin = Perlin3D::new(0);

        // Create vertices of sphere
        let (x_res, y_res) = (100, 50);
        let mut sphere_vertices: Vec<([f32; 3], [f32; 3], [f32; 2])> = Vec::new();
        for x in 0..x_res {
            let azimuth = x as f32 / x_res as f32 * 2.0 * PI;
            for y in 0..=y_res {
                let altitude = y as f32 / y_res as f32 * PI - PI / 2.0;
                let position = Rotation3::from_euler_angles(0.0, altitude, azimuth) * Vector3::from([1.0, 0.0, 0.0]);
                let normal = position.normalize();
                let uv = [azimuth, altitude];
                sphere_vertices.push((position.as_slice().try_into().unwrap(), normal.as_slice().try_into().unwrap(), uv));
            }
        }


        // Create vertex data
        let vertices = sphere_vertices.iter().map(|(v, vn, vt)| {
            // Offset sphere's vertices using noise
            let position = Vector3::from(*v);
            let height =
                0.05 * perlin.sample(position * 55.0)
                + 0.20 * perlin.sample(position * 37.0)
                + 0.75 * perlin.sample(position * 2.0);
            let offset_position = position * (1.0 + 0.3 * height);

            // Store resulting vertex data
            let mut vec = [offset_position.as_slice(), vn].concat();
            vec.extend_from_slice(vt);
            vec
        }).flatten().collect::<Vec<_>>();

        // Create index data
        let mut indices: Vec<u16> = Vec::new();
        // For each quadrilateral in the sphere, indexed by its bottom-left vertex
        for x in 0..x_res {
            for y in 0..y_res {
                let column_length = y_res + 1;
                let bottom_left = x * column_length + y;
                let top_left = bottom_left + 1;
                let bottom_right = if x == x_res - 1 { y } else { bottom_left + column_length };
                let top_right = bottom_right + 1;
                // Output two triangles to make up quad
                indices.extend([bottom_left, top_left, top_right, top_right, bottom_right, bottom_left]);
            }
        }

        let terrain = DynamicBuffer::new(vulkan, &vertices)?;
        let indices = DynamicBuffer::new(vulkan, &indices)?;
        let texture = Texture::new(&vulkan, "res/grid.jpg")?;
        Ok (Asteroid { asteroid_type, size, region: Region::new(size), terrain, indices, texture })
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, view_projection: Matrix4<f32>) {
        unsafe {
            let model = Translation3::from([0.0, 5.0, 0.0]).to_homogeneous();
                // * Rotation3::from_axis_angle(&Vector3::x_axis(), 0.2).to_homogeneous()
                // * Scale3::from([scale, -3.0, scale]).to_homogeneous();
            let data = [model.as_slice(), view_projection.as_slice()].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.texture.descriptor_set], &[]);
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.terrain.buffer], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmdbuf, self.indices.buffer, 0, IndexType::UINT16);
            vulkan.device.cmd_draw_indexed(cmdbuf, self.indices.len, 1, 0, 0, 0);
        }
    }
}

struct Perlin3D {
    seed: usize
}

impl Perlin3D {
    pub fn new(seed: usize) -> Perlin3D {
        Perlin3D { seed }
    }

    fn gradient(&self, x: isize, y: isize, z: isize) -> Vector3<f32> {
        let choices = vec![
            Vector3::from([1.0, 1.0, 0.0]),
            Vector3::from([-1.0, 1.0, 0.0]),
            Vector3::from([1.0, -1.0, 0.0]),
            Vector3::from([-1.0, -1.0, 0.0]),
            Vector3::from([1.0, 0.0, 1.0]),
            Vector3::from([-1.0, 0.0, 1.0]),
            Vector3::from([1.0, 0.0, -1.0]),
            Vector3::from([-1.0, 0.0, -1.0]),
            Vector3::from([0.0, 1.0, 1.0]),
            Vector3::from([0.0, -1.0, 1.0]),
            Vector3::from([0.0, 1.0, -1.0]),
            Vector3::from([0.0, -1.0, -1.0])
        ].iter().map(|v| v.normalize()).collect::<Vec<_>>();
        let mut hasher = hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        x.hash(&mut hasher);
        y.hash(&mut hasher);
        z.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        choices[hash.rem_euclid(choices.len())]
    }

    fn blending(&self, t: f32) -> f32 {
        6.0 * t.powi(5) - 15.0 * t.powi(4) + 10.0 * t.powi(3)
    }

    pub fn sample(&self, position: Vector3<f32>) -> f32 {
        let floor = position.map(|f| f.floor());
        let (floor_x, floor_y, floor_z) = (floor[0] as isize, floor[1] as isize, floor[2] as isize);
        let (u, v, w) = (position[0] - floor[0], position[1] - floor[1], position[2] - floor[2]);
        // For each of the 8 nearest grid points
        let mut contribs = Vec::new();
        for x in floor_x ..= floor_x + 1 {
            for y in floor_y ..= floor_y + 1 {
                for z in floor_z ..= floor_z + 1 {
                    // Get gradient at this grid point
                    let gradient = self.gradient(x, y, z);
                    // Find vector from this grid point to the desired point
                    let vector = Vector3::from([position[0] - x as f32, position[1] - y as f32, position[2] - z as f32]);
                    // Compute contribution
                    contribs.push(gradient.dot(&vector));
                }
                // Blend in z axis
                let a = contribs.pop().unwrap();
                let b = contribs.pop().unwrap();
                let blend = self.blending(w);
                contribs.push(a * blend + b * (1.0 - blend));
            }
            // Blend in y axis
            let a = contribs.pop().unwrap();
            let b = contribs.pop().unwrap();
            let blend = self.blending(v);
            contribs.push(a * blend + b * (1.0 - blend));
        }
        // Blend in x axis
        let a = contribs.pop().unwrap();
        let b = contribs.pop().unwrap();
        let blend = self.blending(u);
        a * blend + b * (1.0 - blend)
    }
}

struct Heightmap {
    width: usize,
    height: usize,
    values: Vec<Vec<f32>>
}

impl Heightmap {
    pub fn new(width: usize, height: usize) -> Heightmap {
        let mut heightmap = Heightmap {
            width, height, values: vec![vec![0.0; height]; width]
        };

        // Initialize corners
        let corner_height = thread_rng().gen_range(0.0..=1.0);
        for (x, y) in [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)] {
            heightmap.values[x][y] = corner_height;
        }

        // Apply diamond-square algorithm
        heightmap.diamond_square(0.65, 1, 0, width - 1, 0, height - 1);

        // Erode terrain
        for _iterations in 0..50 {
            heightmap.erode(-0.02, 0.005);
        }

        // Finally, add roughly gaussian blur
        heightmap.gaussian();

        heightmap
    }

    fn wrap(&self) -> Box<dyn Fn((isize, isize)) -> (usize, usize) + '_> {
        Box::new(move |(x, y)| {
            // Wrap around if one of the vertices is off the edge of the map
            (x.rem_euclid(self.width as isize) as usize, y.rem_euclid(self.height as isize) as usize)
        })
    }

    // Applies gaussian blur to every vertex in the heightmap
    fn gaussian(&mut self) {
        let mut new = vec![vec![0.0; self.height]; self.width];
        // Compute new
        for x in 0..self.width {
            for y in 0..self.height {
                let (x, y) = (x as isize, y as isize);
                let close = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(self.wrap()).map(|(x, y)| self.values[x][y]);
                let far = [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1)].map(self.wrap()).map(|(x, y)| self.values[x][y]);
                let (x, y) = (x as usize, y as usize);
                new[x][y] = self.values[x][y] * 4.0 + close.iter().sum::<f32>() * 2.0 + far.iter().sum::<f32>() * 1.0;
                new[x][y] /= 16.0;
            }
        }
        // Replace map with new
        self.values = new;
    }

    // Applies very basic erosion: if a vertex is higher than its surroundings, it will sacrifice some height to them
    // loss: factor to offset higher vertex by
    // gain: factor of higher vertex height to increase surrounding vertices by
    fn erode(&mut self, loss: f32, gain: f32) {
        let mut delta = vec![vec![0.0; self.height]; self.width];
        // Compute deltas
        for x in 0..self.width {
            for y in 0..self.height {
                let (x, y) = (x as isize, y as isize);
                let surrounding = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(self.wrap());
                let (x, y) = (x as usize, y as usize);
                let max_surrounding = surrounding.iter()
                    .map(|(i, j)| self.values[*i][*j]).max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal)).unwrap();
                if max_surrounding < self.values[x][y] {
                    delta[x][y] += self.values[x][y] * loss;
                    for (i, j) in surrounding {
                        delta[i][j] += self.values[x][y] * gain;
                    }
                }
            }
        }
        // Apply deltas to map
        for x in 0..self.width {
            for y in 0..self.height {
                self.values[x][y] += delta[x][y];
            }
        }
    }

    // Applies the diamond-square algorithm on a heightmap with all 4 corners initialized to the same random value
    // Generates heights between -1.0 and 1.0 which are both x- and y-tileable
    fn diamond_square(&mut self, smoothness: f32, scale: usize, min_x: usize, max_x: usize, min_y: usize, max_y: usize) {
        fn displace(vertex: &mut f32, smoothness: f32, scale: usize) {
            let offset = thread_rng().gen_range(-1.0..=1.0);
            let scaled_offset = offset * (2.0f32.powf(-1.0 * smoothness * scale as f32));
            *vertex += scaled_offset;
        }

        if max_x - min_x < 2 || max_y - min_y < 2 {
            return; // Base case; no inner vertices to update
        }

        // Diamond step: Average 4 corners into center
        let center_x = (max_x + min_x) / 2;
        let center_y = (max_y + min_y) / 2;
        let corners = [self.values[min_x][min_y], self.values[min_x][max_y], self.values[max_x][min_y], self.values[max_x][max_y]];
        self.values[center_x][center_y] = corners.iter().sum::<f32>() / corners.len() as f32;
        // Displace center
        displace(&mut self.values[center_x][center_y], smoothness, scale);

        // Square step: Set midpoints of each edge to average of adjacent vertices
        let half_step = (center_x - min_x) as isize;
        for (x, y) in [(center_x, min_y), (center_x, max_y), (min_x, center_y), (max_x, center_y)] {
            let (x, y) = (x as isize, y as isize);
            // Average 4 adjacents into midpoint
            let corners = [(x - half_step, y), (x + half_step, y), (x, y - half_step), (x, y + half_step)].map(self.wrap());
            self.values[x as usize][y as usize] = corners.iter().map(|(x, y)| self.values[*x][*y]).sum::<f32>() / corners.len() as f32;
            // Displace midpoint
            displace(&mut self.values[x as usize][y as usize], smoothness, scale);
        }

        // Subdivide and recurse
        Heightmap::diamond_square(self, smoothness, scale + 1, min_x, center_x, min_y, center_y);
        Heightmap::diamond_square(self, smoothness, scale + 1, min_x, center_x, center_y, max_y);
        Heightmap::diamond_square(self, smoothness, scale + 1, center_x, max_x, min_y, center_y);
        Heightmap::diamond_square(self, smoothness, scale + 1, center_x, max_x, center_y, max_y);
    }
}
