use std::error::Error;
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
    terrain: DynamicBuffer,
    texture: Texture
}

impl Asteroid {
    pub fn new(vulkan: &VulkanInstance, asteroid_type: AsteroidType, size: [u32; 3]) -> Result<Asteroid, Box<dyn Error>> {
        let heightmap = Heightmap::new(129, 129);

        // Allocate & write vertex buffer
        let mut vertices = Vec::new();
        for x in 0..=heightmap.width - 2 {
            for y in 0..=heightmap.height - 2 {
                // For each square in the heightmap
                let mut gen_triangle_vertices = |a, b, c| {
                    let vs = [a, b, c].map(|(i, j)| {
                        Vector3::from([i as f32, heightmap.values.get::<usize>(i).unwrap()[j], j as f32])
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
        Ok (Asteroid { asteroid_type, size, region: Region::new(size), heightmap, terrain, texture })
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
