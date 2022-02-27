use std::fmt::{Debug};

use nalgebra::{Vector3, vector};

pub struct MarchingCubes {
    cubes: [Cube; 256]
}

// Index corners of a cube as follows:
// 0b000 = min x, y, and z coords
// 0b100 = max x coord
// 0b010 = max y coord
// 0b001 = max z coord
#[derive(Clone, Copy)]
pub struct Cube {
    pattern: u8, // Bit vector of cube corners that are "inside" mesh
    edges: [(u8, u8); 12], // Edges on which this cube outputs a vertex, defined by pairs of cube corners
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
        let mut es = [(0, 0); 12];
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
        let mut es = [(0, 0); 12];
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
        let mut es = [(0, 0); 12];
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

        let mut es = [(0, 0); 12];
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

impl MarchingCubes {
    pub fn new() -> MarchingCubes {
        let basis = [ // 0, 1, 2
            Cube::new(&[], &[]),
            Cube::new(&[CUBE], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z)]),
            Cube::new(&[CUBE, CUBE_X], &[(CUBE, CUBE_Z), (CUBE_X, CUBE_XZ), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Y), (CUBE, CUBE_Z)]),
            // 3, 4, 5
            Cube::new(&[CUBE, CUBE_XY], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_XY, CUBE_XYZ), (CUBE_Y, CUBE_XY)]),
            Cube::new(&[CUBE, CUBE_XYZ], &[(CUBE, CUBE_X), (CUBE, CUBE_Y), (CUBE, CUBE_Z), (CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ)]),
            Cube::new(&[CUBE_X, CUBE_XZ, CUBE_Z], &[(CUBE, CUBE_X), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Z), (CUBE_Z, CUBE_YZ), (CUBE_Z, CUBE_YZ), (CUBE_XZ, CUBE_XYZ), (CUBE_X, CUBE_XY)]),
            // 6, 7, 8
            Cube::new(&[CUBE, CUBE_X, CUBE_XYZ], &[(CUBE_X, CUBE_XZ), (CUBE, CUBE_Z), (CUBE_X, CUBE_XY), (CUBE_X, CUBE_XY), (CUBE, CUBE_Z), (CUBE, CUBE_Y), (CUBE_XZ, CUBE_XYZ), (CUBE_YZ, CUBE_XYZ), (CUBE_XY, CUBE_XYZ)]),
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
        ];

        let rotated: [Cube; 60] = [
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

        MarchingCubes { cubes: (*cubes.as_slice()).try_into().unwrap() }
    }

    fn lookup(&self, pattern: u8) -> Cube {
        self.cubes[pattern as usize]
    }

    // Generates a list of vertices from applying 
    pub fn march(&self, start: Vector3<i32>, end: Vector3<i32>, threshold: f32, noise: Box<dyn Fn(Vector3<f32>) -> f32>) -> Vec<Vector3<f32>> {
        let mut vertices = Vec::new();
        
        // Generate 3D noise texture
        let mut noise_xyz = Vec::new();
        for x in start[0]..=end[0] {
            let mut noise_yz = Vec::new();
            for y in start[1]..=end[1] {
                let mut noise_z = Vec::new();
                for z in start[2]..=end[2] {
                    noise_z.push(noise(vector![x, y, z].cast::<f32>()));
                }
                noise_yz.push(noise_z);
            }
            noise_xyz.push(noise_yz);
        }
        
        for x in start[0]..end[0] {
            for y in start[1]..end[1] {
                for z in start[2]..end[2] {
                    let mut pattern = 0u8;
                    for i in 0..=1 {
                        for j in 0..=1 {
                            for k in 0..=1 {
                                let mut code = 0;
                                if i > 0 { code |= CUBE_X }
                                if j > 0 { code |= CUBE_Y }
                                if k > 0 { code |= CUBE_Z }
                                if noise_xyz[(x - start[0] + i) as usize][(y - start[1] + j) as usize][(z - start[2] + k) as usize] > threshold {
                                    pattern |= 1 << code;
                                }
                            }
                        }
                    }
                    let cube = self.lookup(pattern);
                    let mut corner_pairs = cube.edges.to_vec();
                    corner_pairs.truncate(cube.edges_len as usize);
                    let cube_vertices = corner_pairs.into_iter().map(|(a, b)| {
                        let ax = if a & CUBE_X > 0 { 1 } else { 0 };
                        let ay = if a & CUBE_Y > 0 { 1 } else { 0 };
                        let az = if a & CUBE_Z > 0 { 1 } else { 0 };
                        let bx = if b & CUBE_X > 0 { 1 } else { 0 };
                        let by = if b & CUBE_Y > 0 { 1 } else { 0 };
                        let bz = if b & CUBE_Z > 0 { 1 } else { 0 };
                        let nx = x as f32 + (ax + bx) as f32 / 2.0;
                        let ny = y as f32 + (ay + by) as f32 / 2.0;
                        let nz = z as f32 + (az + bz) as f32 / 2.0;
                        vector![nx, ny, nz]
                    });
                    // Generates vertices of a cube
                    // let v1 = vector![x, y, z];
                    // let v2 = vector![x + 1, y, z];
                    // let v3 = vector![x + 1, y + 1, z];
                    // let v4 = vector![x, y + 1, z];
                    // let v5 = vector![x, y, z + 1];
                    // let v6 = vector![x + 1, y, z + 1];
                    // let v7 = vector![x + 1, y + 1, z + 1];
                    // let v8 = vector![x, y + 1, z + 1];
                    // vec![
                    //     v1, v4, v3,
                    //     v3, v2, v1,
                    //     v2, v3, v7,
                    //     v7, v6, v2,
                    //     v5, v8, v4,
                    //     v4, v1, v5,
                    //     v7, v8, v5,
                    //     v5, v6, v7,
                    //     v4, v8, v7,
                    //     v7, v3, v4,
                    //     v1, v2, v6,
                    //     v6, v5, v1].iter().map(|v| v.cast::<f32>()).collect::<Vec<_>>()
                    vertices.extend(cube_vertices);
                }
            }
        }
        vertices
    }
}
