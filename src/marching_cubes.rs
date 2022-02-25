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

impl Cube {
    fn new(pattern: &[u8], edges: &[(u8, u8)]) -> Cube {
        let mut p = 0u8;
        for corner in pattern {
            p |= 1 << corner;
        }
        let mut es = [(0, 0); 12];
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
        es[..edges.len()].copy_from_slice(&edges);

        Cube { pattern, edges: es, edges_len: self.edges_len }
    }

    fn reflect(&self, axis: u8) -> Cube {
        // For every bit in pattern, reflect it in the direction of axis
        let mut pattern = self.pattern;
        if axis == 0b100 {
            // For x axis: map 76543210 to 32107654
            pattern = ((pattern & 0xf0) >> 4) | ((pattern & 0x0f) << 4);
        } else if axis == 0b010 {
            // For y axis: map 76543210 to 54761032
            pattern = ((pattern & 0xcc) >> 2) | ((pattern & 0x33) << 2);
        } else if axis == 0b001 {
            // For z axis: map 76543210 to 67452301
            pattern = ((pattern & 0xaa) >> 1) | ((pattern & 0x55) << 1);
        }
        let edges = self.edges
            .map(|(v1, v2)| (v1 ^ axis, v2 ^ axis))
            .chunks(3)
            .map(|tri| [tri[0], tri[2], tri[1]])
            .flatten()
            .collect::<Vec<_>>();
        let mut es = [(0, 0); 12];
        es[..edges.len()].copy_from_slice(&edges);
        Cube { pattern, edges: es, edges_len: self.edges_len }
    }

    fn rotate(&self, axis: u8) -> Cube {
        let mut pattern = self.pattern;
        let mut edges = self.edges;

        if axis & 0b001 > 0 {
            // Rotate 90 degrees clockwise about the Z axis
            // CUBE -> CUBE_Y -> CUBE_XY -> CUBE_X
            // Map 76543210 to 54107632
            // 76 shr 4, 54 shl 2, 32 shr 2, 10 shl 4
            pattern = ((self.pattern & 0xc0) >> 4) | ((self.pattern & 0x30) << 2)
                    | ((self.pattern & 0x0c) >> 2) | ((self.pattern & 0x03) << 4);
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
        if axis & 0b010 > 0 {
            // Rotate 90 degrees clockwise about the Y axis
            // CUBE -> CUBE_X -> CUBE_XZ -> CUBE_Z
            // Map 76543210 to 37152604
            // 7_5 >> 1, 6_4 >> 4, 3_1 << 4, 2_0 << 1
            pattern = ((self.pattern & 0xa0) >> 1) | ((self.pattern & 0x50) >> 4)
                    | ((self.pattern & 0x0a) << 4) | ((self.pattern & 0x05) << 1);
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
        if axis & 0b100 > 0 {
            // Rotate 90 degrees clockwise about the X axis
            // CUBE -> CUBE_Z -> CUBE_YZ -> CUBE_Y
            // Map 76543210 to 64752031
            // 7___3 >> 2, 6___2 << 1, 5___1 >> 1, 4___0 << 2
            pattern = ((self.pattern & 0x88) >> 2) | ((self.pattern & 0x44) << 1)
                    | ((self.pattern & 0x22) >> 1) | ((self.pattern & 0x11) << 2);
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
            Cube::new(&[CUBE, CUBE_XYZ], &[]),
            Cube::new(&[CUBE_X, CUBE_XZ, CUBE_Z], &[]),
            // 6, 7, 8
            Cube::new(&[CUBE, CUBE_X, CUBE_XYZ], &[]),
            Cube::new(&[CUBE_X, CUBE_Y, CUBE_XYZ], &[]),
            Cube::new(&[CUBE, CUBE_X, CUBE_Z, CUBE_XZ], &[]),
            // 9, 10, 11
            Cube::new(&[CUBE, CUBE_Z, CUBE_XZ, CUBE_YZ], &[]),
            Cube::new(&[CUBE, CUBE_Y, CUBE_XZ, CUBE_XYZ], &[]),
            Cube::new(&[CUBE, CUBE_Z, CUBE_XZ, CUBE_XYZ], &[]),
            // 12, 13, 14
            Cube::new(&[CUBE_X, CUBE_Y, CUBE_Z, CUBE_XZ], &[]),
            Cube::new(&[CUBE, CUBE_XZ, CUBE_XY, CUBE_YZ], &[]),
            Cube::new(&[CUBE_X, CUBE_Z, CUBE_XZ, CUBE_YZ], &[])
        ];

        let rotated: [Cube; 120] = [
                basis,
                basis.map(|c| c.rotate(0b001)),
                basis.map(|c| c.rotate(0b010)),
                basis.map(|c| c.rotate(0b011)),
                basis.map(|c| c.rotate(0b100)),
                basis.map(|c| c.rotate(0b101)),
                basis.map(|c| c.rotate(0b110)),
                basis.map(|c| c.rotate(0b111))
            ].concat().as_slice().try_into().unwrap();
        let reflected_x = rotated.map(|c| c.reflect(0b100));
        let reflected_y = rotated.map(|c| c.reflect(0b010));
        let reflected_z = rotated.map(|c| c.reflect(0b001));
        let reflected_xy = reflected_x.map(|c| c.reflect(0b010));
        let reflected_xz = reflected_x.map(|c| c.reflect(0b001));
        let reflected_yz = reflected_y.map(|c| c.reflect(0b001));
        let reflected_xyz = reflected_xy.map(|c| c.reflect(0b001));
        let reflected = [rotated, reflected_x, reflected_y, reflected_z, reflected_xy, reflected_xz, reflected_yz, reflected_xyz].concat();
        let inverted = reflected.iter().map(|c| c.invert());
        let mut all = Vec::new();
        all.extend(&reflected);
        all.extend(inverted);
        dbg!(all.len());
        all.sort_by_key(|c| c.pattern);
        all.dedup_by_key(|c| c.pattern);
        dbg!(all.len());

        let mut cubes = [basis[0]; 256];
        for cube in all {
            cubes[cube.pattern as usize] = cube;
        }
        MarchingCubes { cubes }
    }

    pub fn lookup(&self, pattern: u8) -> Cube {
        self.cubes[pattern as usize]
    }
}
