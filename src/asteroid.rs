use std::error::Error;
use std::slice;
use std::cmp::Ordering::Equal;

use ash::vk::{CommandBuffer, BufferUsageFlags, MemoryMapFlags, Buffer, ShaderStageFlags, PipelineBindPoint};
use nalgebra::{Matrix4, Vector3, Translation3, Scale3};
use rand::{thread_rng, Rng};

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
    terrain: Buffer,
    terrain_len: usize,
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
        let (vertex_buffer, _vertex_allocation) = unsafe {
            let (buffer, allocation) = vulkan.allocator.allocate_buffer(&vulkan.device, BufferUsageFlags::VERTEX_BUFFER, (vertices.len() * 4) as u64)?;
            let dst = vulkan.device.map_memory(allocation.memory, allocation.offset, allocation.size, MemoryMapFlags::empty())?;
            (dst as *mut f32).copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            vulkan.device.unmap_memory(allocation.memory);
            (buffer, allocation)
        };

        let texture = Texture::new(&vulkan, "res/grid.jpg")?;

        Ok (Asteroid { asteroid_type, size, region: Region::new(size), heightmap, terrain: vertex_buffer, terrain_len: vertices.len(), texture })
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
            vulkan.device.cmd_bind_vertex_buffers(cmdbuf, 0, &[self.terrain], &[0]);
            vulkan.device.cmd_draw(cmdbuf, self.terrain_len as u32, 1, 0, 0)
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
        let mut map = vec![vec![0.0; height]; width];

        fn displace(map: &mut f32, scale: usize) {
            let offset = thread_rng().gen_range(-1.0..=1.0);
            let scaled_offset = offset * (2.0f32.powf(-0.65 * scale as f32));
            *map += scaled_offset;
        }

        fn diamond_square(map: &mut Vec<Vec<f32>>, scale: usize, min_x: usize, max_x: usize, min_y: usize, max_y: usize) {
            if max_x - min_x < 2 || max_y - min_y < 2 {
                return; // Base case; no inner vertices to update
            }

            // Diamond step: Average 4 corners into center
            let center_x = (max_x + min_x) / 2;
            let center_y = (max_y + min_y) / 2;
            let corners = [map[min_x][min_y], map[min_x][max_y], map[max_x][min_y], map[max_x][max_y]];
            map[center_x][center_y] = corners.iter().sum::<f32>() / corners.len() as f32;
            // Displace center
            displace(&mut map[center_x][center_y], scale);

            // Square step: Set midpoints of each edge to average of adjacent vertices
            let half_step = (center_x - min_x) as isize;
            let width = map.len() as isize;
            let height = map[0].len() as isize;
            for (x, y) in [(center_x, min_y), (center_x, max_y), (min_x, center_y), (max_x, center_y)].map(|(x, y)| (x as isize, y as isize)) {
                // Average 4 adjacents into midpoint
                let corners = [(x - half_step, y), (x + half_step, y), (x, y - half_step), (x, y + half_step)].map(|(x, y)| {
                    (x.rem_euclid(width) as usize, y.rem_euclid(height) as usize) // Wrap around if one of the vertices is off the edge of the map
                });
                map[x as usize][y as usize] = corners.iter().map(|(x, y)| map[*x][*y]).sum::<f32>() / corners.len() as f32;
                // Displace midpoint
                displace(&mut map[x as usize][y as usize], scale);
            }

            // Subdivide and recurse
            diamond_square(map, scale + 1, min_x, center_x, min_y, center_y);
            diamond_square(map, scale + 1, min_x, center_x, center_y, max_y);
            diamond_square(map, scale + 1, center_x, max_x, min_y, center_y);
            diamond_square(map, scale + 1, center_x, max_x, center_y, max_y);
        }

        // Initialize corners
        displace(&mut map[0][0], 0);
        for (x, y) in [(0, height - 1), (width - 1, 0), (width - 1, height - 1)] {
            map[x][y] = map[0][0];
        }

        // Apply diamond-square algorithm
        diamond_square(&mut map, 1, 0, width - 1, 0, height - 1);

        // Apply very basic erosion: if a vertex is higher than its surroundings, it will sacrifice some height to them
        let loss = -0.02; // Factor to decrease higher vertex by
        let gain = 0.005; // Factor to increase surrounding vertices by
        for _iterations in 0..50 {
            let mut delta = vec![vec![0.0; height]; width];

            // Compute deltas
            for x in 1 ..= width - 2 {
                for y in 1 ..= height - 2 {
                    let surrounding = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)];
                    let max_surrounding = surrounding.iter()
                        .map(|(i, j)| map[*i][*j]).max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal)).unwrap();
                    if max_surrounding < map[x][y] {
                        delta[x][y] += map[x][y] * loss;
                        for (i, j) in surrounding {
                            delta[i][j] += map[x][y] * gain;
                        }
                    }
                }
            }

            // Apply deltas to map
            for x in 0..width {
                for y in 0..height {
                    map[x][y] += delta[x][y];
                }
            }
        }

        // Apply roughly gaussian blur
        for _iterations in 0..1 {
            let mut new = vec![vec![0.0; height]; width];

            // Compute new
            for x in 0..width {
                for y in 0..height {
                    let close = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].map(|(x, y)| {
                        // Wrap around if one of the vertices is off the edge of the map
                        ((x as isize).rem_euclid(width as isize) as usize, (y as isize).rem_euclid(height as isize) as usize)
                    }).map(|(x, y)| map[x][y]);
                    let far = [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1)].map(|(x, y)| {
                        ((x as isize).rem_euclid(width as isize) as usize, (y as isize).rem_euclid(height as isize) as usize)
                    }).map(|(x, y)| map[x][y]);
                    new[x][y] = map[x][y] * 4.0 + close.iter().sum::<f32>() * 2.0 + far.iter().sum::<f32>() * 1.0;
                    new[x][y] /= 16.0;
                }
            }

            // Replace map with new
            map = new;
        }

        // for j in 0..height {
        //     for i in 0..width {
        //         print!("{:.2} ", map[i][j]);
        //     }
        //     println!();
        // }

        Heightmap { width, height, values: map }
    }
}
