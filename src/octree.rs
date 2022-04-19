use std::collections::BinaryHeap;

use nalgebra::{Vector3, vector};

use crate::rtree::BoundingBox;

pub enum Octree {
    Branch {
        center: Vector3<f32>,
        children: [Option<Box<Octree>>; 8],
    },
    Point (Vector3<f32>, u32)
}

impl Default for Octree {
    fn default() -> Self {
        Self::Point (Vector3::zeros(), Default::default())
    }
}

struct NearestNeighborState<'a> {
    node: &'a Octree, 
    bounding_box: BoundingBox,
    distance: f32
}

impl<'a> Eq for NearestNeighborState<'a> {}

impl<'a> PartialEq for NearestNeighborState<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl<'a> Ord for NearestNeighborState<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.partial_cmp(&self.distance).expect("Ord for NearestNeighborState failed!")
    }
}

impl<'a> PartialOrd for NearestNeighborState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some (self.cmp(&other))
    }
}

enum ExploreState {
    ExploringUp, ExploringDown
}

impl Octree {
    pub fn new(points: &mut dyn ExactSizeIterator<Item = Vector3<f32>>) -> Octree {
        let mut work_queue = Vec::new();
        let mut root = Default::default();
        let indexed_points: Vec<(Vector3<f32>, u32)> = points.filter(|v| !v[0].is_nan()).zip(0..).collect::<Vec<_>>(); // Filter out NaN's which are empty clusters
        work_queue.push((&mut root, indexed_points));
        while let Some ((parent, points)) = work_queue.pop() {
            if points.len() == 1 {
                // Leaf node
                *parent = Self::Point(points[0].0, points[0].1);
            } else {
                // Branch node
                let todo_len = points.len();
                let center = Octree::center((Box::new(points.iter().map(|p| &p.0)) as Box<dyn ExactSizeIterator<Item = &Vector3<f32>>>).as_mut());
                let mut child_points: [Option<Vec<(Vector3<f32>, u32)>>; 8] = Default::default();
                for i in 0..8 { // Initialize empty vecs in child_points
                    child_points[i] = Some (Vec::new());
                }
                for p in points.iter() {
                    let octant = Octree::octant(center, p.0);
                    child_points.get_mut(octant as usize).unwrap().as_mut().unwrap().push(*p);
                }
                let mut children: [Option<Box<Octree>>; 8] = Default::default();
                let mut num_octants = 0;
                for i in 0..8 {
                    if child_points.get(i).unwrap().as_ref().unwrap().is_empty() {
                        children[i] = None;
                    } else {
                        num_octants += 1;
                        children[i] = Some (Box::new(Default::default()));
                    }
                }
                if num_octants == 1 {
                    // All points in this subtree are identical
                    println!("center: {:?} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", center);
                    println!("len: {}", todo_len);
                    *parent = Self::Point(points[0].0, points[0].1);
                } else {
                    *parent = Self::Branch { center, children };
                    if let Self::Branch { children, .. } = parent {
                        for (i, subtree) in children.iter_mut().enumerate() {
                            if let Some (child) = subtree.as_mut() {
                                work_queue.push((child.as_mut(), child_points[i].take().unwrap()));
                            }
                        }
                    }
                }
            }
        }
        root
    }

    pub fn size(&self) -> usize {
        match self {
            Octree::Point (..) => 1,
            Octree::Branch { children, .. } => {
                let mut size = 0;
                for child in children {
                    if let Some (c) = child {
                        size += c.size();
                    }
                }
                size
            }
        }
    }

    // Returns the point in the octree which is closest to the query point
    pub fn nearest_neighbor(&self, query: Vector3<f32>) -> (Vector3<f32>, f32, u32) {
        let initial_box = BoundingBox::from_point(vector![
            f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
            f32::INFINITY, f32::INFINITY, f32::INFINITY]);
        /*let mut priority_queue = BinaryHeap::new();
        priority_queue.push(NearestNeighborState {
            node: self,
            bounding_box: initial_box,
            distance: 0.0,
        });

        while let Some (NearestNeighborState { node, bounding_box, distance }) = priority_queue.pop() {
            match node {
                Self::Point (center, index) => { // Because priority queue ensures we see closest points first, the first point we see is the nearest neighbor
                    return (*center, distance, *index);
                },
                Self::Branch { center, children } => {
                    for i in 0..8 {
                        if let Some (child) = children.get(i as usize).unwrap() {
                            let child_box = {
                                let mut new_bounds = bounding_box.point();
                                new_bounds[3 - 3 * ((i & 0b100) >> 2) as usize] = center[0];
                                new_bounds[4 - 3 * ((i & 0b010) >> 1) as usize] = center[1];
                                new_bounds[5 - 3 * (i & 0b001) as usize] = center[2];
                                BoundingBox::from_point(new_bounds)
                            };
                            let distance = match child.as_ref() {
                                Self::Point (center, _) => center.metric_distance(&query),
                                Self::Branch {..} => child_box.point_distance(query)
                            };
                            priority_queue.push(NearestNeighborState {
                                node: child.as_ref(),
                                bounding_box: child_box,
                                distance
                            });
                        }
                    }
                }
            }
        }

        todo!() // TODO return an option i guess?*/


        let mut subtrees = vec![(self, initial_box, ExploreState::ExploringDown)];
        let mut best = None;
        let mut best_index = u32::MAX;
        let mut best_distance = f32::INFINITY;
        while let Some ((stree, bounding_box, explore_state)) = subtrees.last() {
            match stree {
                Self::Point (point, idx) => {
                    let point_distance = (query - point).magnitude();
                    if point_distance < best_distance {
                        best = Some (point);
                        best_index = *idx;
                        best_distance = point_distance;
                    }
                    subtrees.pop(); // Done with this leaf node
                    if let Some (parent) = subtrees.last_mut() {
                        parent.2 = ExploreState::ExploringUp; // If not last node, return to parent and start going back up
                    }
                },
                Self::Branch { center, children } => {
                    let octant = Octree::octant(*center, query);
                    match explore_state {
                        ExploreState::ExploringDown => {
                            // First, explore down octree to find smallest node containing query to find initial result
                            match children.get(octant as usize).unwrap() {
                                Some (subtree) => {
                                    let mut new_bounds = bounding_box.point(); // TODO factor out
                                    new_bounds[3 - 3 * ((octant & 0b100) >> 2) as usize] = center[0];
                                    new_bounds[4 - 3 * ((octant & 0b010) >> 1) as usize] = center[1];
                                    new_bounds[5 - 3 * (octant & 0b001) as usize] = center[2];
                                    subtrees.push((subtree.as_ref(), BoundingBox::from_point(new_bounds), ExploreState::ExploringDown));
                                },
                                None => {
                                    subtrees.last_mut().unwrap().2 = ExploreState::ExploringUp;
                                }
                            }
                        },
                        ExploreState::ExploringUp => {
                            // Next, return up the tree, checking which other nodes may contain better results
                            // Search them & update best result distance accordingly
                            let current_bounds = bounding_box.point();
                            subtrees.pop(); // Remove this node. We'll either add its other children, or return further up
                            if let Some ((_, _, parent_state)) = subtrees.last_mut() {
                                *parent_state = ExploreState::ExploringUp; // If we have a parent, set it to also return further up
                            }
                            for i in 0..8 {
                                if let Some (sibling) = children.get(i as usize).unwrap() {
                                    // Sibling subtree
                                    let sibling_box = {
                                        let mut new_bounds = current_bounds.clone();
                                        new_bounds[3 - 3 * ((i & 0b100) >> 2) as usize] = center[0];
                                        new_bounds[4 - 3 * ((i & 0b010) >> 1) as usize] = center[1];
                                        new_bounds[5 - 3 * (i & 0b001) as usize] = center[2];
                                        BoundingBox::from_point(new_bounds)
                                    };
                                    let possible_distance = sibling_box.point_distance(query);
                                    if possible_distance != 0.0 && possible_distance < best_distance {
                                        // Explore this subtree later
                                        subtrees.push((sibling.as_ref(), sibling_box, ExploreState::ExploringDown));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (*best.unwrap(), best_distance, best_index)
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

    fn center(vertices: &mut dyn ExactSizeIterator<Item = &Vector3<f32>>) -> Vector3<f32> {
        let len = vertices.len() as f32;
        let sum = vertices.into_iter().fold(Vector3::zeros(), |acc, vtx| acc + vtx);
        sum.component_div(&vector![len, len, len])
    }
}
