use std::fmt::Debug;
use std::rc::Rc;

use nalgebra::{Vector3, vector};

pub struct Octree<I> {
    pub center: Vector3<f32>,
    pub children: [Option<Box<Octree<I>>>; 8],
    pub triangles: Vec<I>,
    pub size: usize
}

impl<I> Octree<I> where I: Copy + TryInto<usize>, <I as TryInto<usize>>::Error: Debug {
    pub fn new(vertices: Rc<Vec<Vector3<f32>>>, indices: &Vec<I>) -> Octree<I> {
        let mut work_queue = Vec::new();
        let mut root = Octree {
            center: Octree::center(vertices.clone(), indices),
            children: Default::default(),
            triangles: indices.iter().map(|i| *i).collect::<Vec<_>>(), 
            size: indices.len()
        };
        dbg!(root.size);
        work_queue.push(&mut root);

        let mut children = 0;
        let mut max_triangles = 0;
        while let Some (octree) = work_queue.pop() {
            let mut child_triangles: [Vec<I>; 8] = Default::default();
            if octree.triangles.len() > 128 { // If node is large enough to warrant children
                // Attempt to subdivide
                let mut triangles = Vec::new();
                for i in (0..octree.triangles.len()).step_by(3) {
                    let tri = [octree.triangles[i], octree.triangles[i + 1], octree.triangles[i + 2]];
                    let (a, b, c) = (vertices[tri[0].try_into().unwrap()], vertices[tri[1].try_into().unwrap()], vertices[tri[2].try_into().unwrap()]);
                    let (qa, qb, qc) = (Octree::<I>::octant(octree.center, a), Octree::<I>::octant(octree.center, b), Octree::<I>::octant(octree.center, c));
                    if qa == qb && qb == qc {
                        // Triangle is entirely within one octant; add it there
                        child_triangles[qa as usize].extend(tri);
                    } else {
                        // Vertices of triangle are in different octants; cannot live in any single child
                        triangles.extend(tri);
                    }
                }
                octree.triangles = triangles; // Triangles in multipe octants remain in parent
                max_triangles = max_triangles.max(octree.triangles.len());

                // Generate child octants
                for (i, child_tris) in child_triangles.into_iter().enumerate() {
                    if child_tris.len() > 0 { // If a child node is necessary
                        let size = child_tris.len();
                        let child = Box::new(Octree { center: Octree::center(vertices.clone(), &child_tris), children: Default::default(), triangles: child_tris, size });
                        children += 1;
                        if children % 10_000 == 0 {
                            dbg!(children);
                        }
                        octree.children[i] = Some (child);
                    }
                }
                // ... and schedule them to be subdivided
                for child in octree.children.iter_mut() {
                    if let Some (c) = child.as_mut() {
                        work_queue.push(c.as_mut());
                    }
                }
            }
        }
        dbg!(children, max_triangles);
        root
    }

    // Returns a bit vector with 0b100 is vertex.x >= center.x, and similarly 0b010 and 0b001 for y and z respectively
    pub fn octant(center: Vector3<f32>, vertex: Vector3<f32>) -> u8 {
        let mut result = 0;
        if vertex[0] >= center[0] {
            result |= 0b100;
        }
        if vertex[1] >= center[1] {
            result |= 0b10;
        }
        if vertex[2] >= center[2] {
            result |= 0b1;
        }
        result
    }

    fn center(vertices: Rc<Vec<Vector3<f32>>>, indices: &Vec<I>) -> Vector3<f32> {
        let mut used_indices = vec![false; vertices.len()];
        for i in indices {
            used_indices[(*i).try_into().unwrap()] = true;
        }
        let used_vertices = vertices.iter().enumerate()
            .filter_map(|(i, vtx)| if used_indices[i] { Some (vtx) } else { None })
            .collect::<Vec<_>>();
        let used_count = used_vertices.len();
        let sum = used_vertices.into_iter().fold(Vector3::zeros(), |acc, vtx| acc + vtx);
        let avg = sum.component_div(&vector![used_count, used_count, used_count].cast::<f32>());
        avg.map(|f| f.round())
    }
}
