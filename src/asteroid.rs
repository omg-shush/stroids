use std::collections::hash_map;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::slice;
use std::cmp::Ordering::Equal;
use std::convert::From;

use ash::vk::{CommandBuffer, ShaderStageFlags, PipelineBindPoint};
use nalgebra::{Matrix4, Vector3, Translation3, Scale3};
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
    heightmap: Heightmap,
    perlin: Perlin3D,
    terrain: DynamicBuffer,
    texture: Texture
}

impl Asteroid {
    pub fn new(vulkan: &VulkanInstance, asteroid_type: AsteroidType, size: [u32; 3]) -> Result<Asteroid, Box<dyn Error>> {
        let heightmap = Heightmap::new(129, 129);
        let perlin = Perlin3D::new(0);

        // Allocate & write vertex buffer
        let mut vertices = Vec::new();
        for x in 0..=heightmap.width - 2 {
            for y in 0..=heightmap.height - 2 {
                // For each square in the heightmap
                let mut gen_triangle_vertices = |a, b, c| {
                    let vs = [a, b, c].map(|(i, j)| {
                        // Vector3::from([i as f32, heightmap.values.get::<usize>(i).unwrap()[j], j as f32])
                        let height =
                            0.25 * perlin.sample(Vector3::from([i as f32 / 8.0, j as f32 / 8.0, 0.3]))
                            + 0.75 * perlin.sample(Vector3::from([i as f32 / 32.0, j as f32 / 32.0, 0.7]));
                        Vector3::from([i as f32, height, j as f32])
                    });
                    let vn = (vs[2] - vs[0]).cross(&(vs[1] - vs[0])).normalize();

                    for vertex in vs {
                        vertices.extend_from_slice(vertex.as_slice());
                        vertices.extend_from_slice(vn.as_slice());
                        vertices.extend([vertex[0] / heightmap.width as f32, vertex[2] / heightmap.height as f32]); // UV coordinate
                    }
                };
                
                gen_triangle_vertices((x, y), (x + 1, y), (x + 1, y + 1));
                gen_triangle_vertices((x + 1, y + 1), (x, y + 1), (x, y));
            }
        }
        let terrain = DynamicBuffer::new(vulkan, &vertices)?;
        let texture = Texture::new(&vulkan, "res/grid.jpg")?;
        Ok (Asteroid { asteroid_type, size, region: Region::new(size), heightmap, perlin, terrain, texture })
    }

    pub fn render(&self, vulkan: &VulkanInstance, cmdbuf: CommandBuffer, view_projection: Matrix4<f32>) {
        unsafe {
            let scale = 10.0;
            let model = Translation3::from([scale / -2.0, 6.0, scale / -2.0]).to_homogeneous()
                // * Rotation3::from_axis_angle(&Vector3::x_axis(), 0.2).to_homogeneous()
                * Scale3::from([scale / self.heightmap.width as f32, -3.0, scale / self.heightmap.height as f32]).to_homogeneous();
            let data = [model.as_slice(), view_projection.as_slice()].concat();
            let bytes = slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            vulkan.device.cmd_push_constants(cmdbuf, vulkan.pipeline_layout, ShaderStageFlags::VERTEX, 0, bytes);

            vulkan.device.cmd_bind_descriptor_sets(cmdbuf, PipelineBindPoint::GRAPHICS, vulkan.pipeline_layout,
                1, &[self.texture.descriptor_set], &[]);
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.terrain.buffer], &[0]);
            vulkan.device.cmd_draw(cmdbuf, self.terrain.len, 1, 0, 0)
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
