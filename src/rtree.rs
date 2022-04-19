use std::rc::Rc;
use std::fmt::Debug;

use nalgebra::{Vector3, Vector6, vector};
use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::octree::Octree;

pub enum RTree<I> {
    Branch (Vec<(BoundingBox, Box<RTree<I>>)>),
    Leaf (Vec<I>)
}

impl<I> RTree<I> where
    I: Copy + TryInto<usize> + TryFrom<usize>,
    <I as TryInto<usize>>::Error: Debug,
    <I as TryFrom<usize>>::Error: Debug {
    pub fn new(vertices: Rc<Vec<Vector3<f32>>>, indices: &Vec<I>) -> RTree<I> {
        if indices.len() < 128 * 3 {
            Self::Leaf(indices.to_owned())
        } else {
            let bounding_boxes = indices.chunks_exact(3).map(|tri| {
                let a = vertices[tri[0].try_into().unwrap()];
                let b = vertices[tri[1].try_into().unwrap()];
                let c = vertices[tri[2].try_into().unwrap()];
                BoundingBox::from(&[a, b, c])
            }).collect::<Vec<_>>();
            let rectangles = RTree::<I>::kmeans(&bounding_boxes, bounding_boxes.len() / 128);
            let mut subtrees = rectangles.into_iter().map(|r| {
                let leaf_indices = r.into_iter().flat_map(|i| {
                    let i = i.try_into().unwrap();
                    indices[i * 3..i * 3 + 3].into_iter()
                }).map(|i| *i).collect::<Vec<I>>();
                Some (Self::Leaf(leaf_indices))
            }).collect::<Vec<_>>();
            while subtrees.len() > 1 {
                let bounding_boxes = subtrees.iter().map(|s| s.as_ref().unwrap().bounding_box(vertices.clone())).collect::<Vec<_>>();
                let rectangles = RTree::<I>::kmeans(&bounding_boxes, bounding_boxes.len() / 128 + 1);
                subtrees = rectangles.iter().map(|cluster| {
                    let children = cluster.iter().map(|i| {
                        let rtree = subtrees[(*i).try_into().unwrap()].take().unwrap();
                        let bounding = rtree.bounding_box(vertices.clone());
                        (bounding, Box::new(rtree))
                    }).collect::<Vec<_>>();
                    Some (Self::Branch(children))
                }).collect::<Vec<_>>();
            }
            subtrees.remove(0).unwrap()
        }
    }

    pub fn bounding_box(&self, vertices: Rc<Vec<Vector3<f32>>>) -> BoundingBox {
        match self {
            Self::Branch (children) => BoundingBox::from_iter(children.iter().map(|c| c.0)),
            Self::Leaf (indices) => {
                BoundingBox::from_iter(indices.chunks_exact(3).map(|tri| {
                    let a = vertices[tri[0].try_into().unwrap()];
                    let b = vertices[tri[1].try_into().unwrap()];
                    let c = vertices[tri[2].try_into().unwrap()];
                    BoundingBox::from(&[a, b, c])
                }))
            }
        }
    }

    // Treats each bounding box as a 6-dimensional point
    pub fn kmeans(bounding_boxes: &[BoundingBox], k: usize) -> Vec<Vec<I>> {
        if k == 1 {
            vec![(0..bounding_boxes.len()).map(|i| i.try_into().unwrap()).collect::<Vec<_>>()]
        } else {
            let mut cluster_members: Vec<Vec<I>> = Vec::new();
            for _ in 0..k {
                cluster_members.push(Vec::new());
            }

            // Select k random points as initial cluster centers
            let mut cluster_centers = bounding_boxes.choose_multiple(&mut thread_rng(), k).map(|x| *x).collect::<Vec<_>>();

            let mut max_delta = f32::INFINITY;
            let mut iterations = 0;
            while max_delta > 1.0 {
                // Reset cluster members
                for cluster in cluster_members.iter_mut() {
                    cluster.clear();
                }
                // Create octree of cluster centers
                let center_points = cluster_centers.iter().map(|bb| (bb.min + bb.max).component_div(&vector![2.0, 2.0, 2.0])); // TODO make octree take Vector6?
                println!("center_points: {}", center_points.len());
                let octree_centers = Octree::new((Box::new(center_points) as Box<dyn ExactSizeIterator<Item = Vector3<f32>>>).as_mut());
                println!("octree_centers: {}", octree_centers.size());
                // Update cluster assignments
                let mut discrepancy = 0;
                for (i, bb) in bounding_boxes.iter().enumerate() {
                    let point = bb.point();
                    let point = (bb.min + bb.max).component_div(&vector![2.0, 2.0, 2.0]);
                    // Find closest cluster of point
                    let (_, distance_octree, closest_center_octree) = octree_centers.nearest_neighbor(point); // TODO Vector6 point?
                    // TODO accelerate nearest-center search using octree maybe?
                    /*let mut closest_center = usize::MAX;
                    let mut closest_dist = f32::INFINITY;
                    for (j, center) in cluster_centers.iter().enumerate() {
                        let dist = (center.point() - point).magnitude_squared();
                        if dist < closest_dist {
                            closest_dist = dist;
                            closest_center = j;
                        }
                    }*/
                    let mut closest_center = usize::MAX;
                    let mut closest_dist = f32::INFINITY;
                    for (j, center) in cluster_centers.iter().enumerate() {
                        let center_point = (center.min + center.max).component_div(&vector![2.0, 2.0, 2.0]);
                        let dist = (center_point - point).magnitude_squared();
                        if dist < closest_dist {
                            closest_dist = dist;
                            closest_center = j;
                        }
                    }
                    if closest_center_octree as usize != closest_center {
                        discrepancy += 1;
                        // println!("is {}, should be {}", closest_center_octree, closest_center);
                        // println!("dist is {}, should be {}", distance_octree, closest_dist);
                        // panic!("ERROR! not equal!!!");
                    }
                    // Assign point to closest cluster
                    cluster_members[closest_center as usize].push(i.try_into().unwrap());
                }
                // Update cluster centers
                let mut new_max_delta = 0.0f32;
                let mut nonempty_clusters = 0;
                for i in 0..k {
                    if !cluster_members[i].is_empty() {
                        nonempty_clusters += 1;
                    }
                    let mut sum = Vector6::zeros();
                    for bb in cluster_members[i].iter() {
                        sum += bounding_boxes[(*bb).try_into().unwrap()].point();
                    }
                    let new_center = BoundingBox::from_point(sum / (cluster_members[i].len() as f32));
                    new_max_delta = new_max_delta.max((cluster_centers[i].point() - new_center.point()).magnitude_squared());
                    cluster_centers[i] = new_center;
                }
                iterations += 1;
                println!("iteration {}, with {} clusters, had max delta {}, with {} discrepancies", iterations, nonempty_clusters, new_max_delta, discrepancy);
                max_delta = new_max_delta;
            }
            println!("kmeans iterations: {}", iterations);
            
            // Remove empty clusters
            cluster_members.into_iter().filter(|vec| !vec.is_empty()).collect::<Vec<_>>()
            // cluster_members.drain_filter(|vec| vec.is_empty()).last(); // TODO unstable feature
        }
    }
}

#[derive(Clone, Copy)]
pub struct BoundingBox {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>
}

impl BoundingBox {
    pub fn from(vertices: &[Vector3<f32>]) -> BoundingBox {
        let (mut inf, mut sup) = (vertices[0], vertices[0]);
        for i in 0..vertices.len() {
            inf = inf.inf(&vertices[i]);
            sup = sup.sup(&vertices[i]);
        }
        BoundingBox { min: inf, max: sup }
    }

    // Requires boxes is nonempty
    pub fn from_iter(mut boxes: impl Iterator<Item = BoundingBox>) -> BoundingBox {
        let first = boxes.next().unwrap();
        let (mut inf, mut sup) = (first.min, first.max);
        while let Some (item) = boxes.next() {
            inf = inf.inf(&item.min);
            sup = sup.sup(&item.max);
        }
        BoundingBox { min: inf, max: sup }
    }

    pub fn from_point(p: Vector6<f32>) -> BoundingBox {
        let s = p.as_slice();
        let min: [f32; 3] = s[0..3].try_into().unwrap();
        let max: [f32; 3] = s[3..6].try_into().unwrap();
        BoundingBox { min: Vector3::from(min), max: Vector3::from(max) }
    }

    pub fn point(&self) -> Vector6<f32> {
        let x = self.min.iter().chain(self.max.iter()).map(|f| *f).collect::<Vec<_>>();
        Vector6::from_column_slice(&x)
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        let mut result = true;
        for i in 0..3 {
            result &= self.min[i] < other.max[i] && other.min[i] < self.max[i];
        }
        result
    }

    pub fn point_distance(&self, point: Vector3<f32>) -> f32 {
        let mut distance_squared = 0.0;
        for i in 0..3 {
            if point[i] < self.min[i] {
                distance_squared += (self.min[i] - point[i]).powi(2);
            } else if point[i] > self.max[i] {
                distance_squared += (point[i] - self.max[i]).powi(2);
            }
        }
        distance_squared.sqrt()
    }
}
