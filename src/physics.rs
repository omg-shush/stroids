use std::mem::swap;
use std::rc::Rc;

use nalgebra::{Vector3, SVD, Matrix3x2, Matrix3, UnitQuaternion, Point3, Translation3, Scale3, Matrix4};

pub type Entity = usize;

pub struct PhysicsEngine {
    entity_count: usize,
    entity: Vec<EntityState>,
    entity_properties: Vec<EntityProperties>
}

pub struct EntityProperties {
    pub immovable: bool, // If this object should move due to simulation
    pub collision: bool, // If other objects should collide with this
    pub gravitational: bool // If other objects should experience attraction to this
}

pub struct EntityState {
    pub prev_position: Vector3<f32>,
    pub position: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub velocity: Vector3<f32>,
    pub vertices: Rc<Vec<Vector3<f32>>>,
    pub mass: f32,
    pub mesh: Vec<Mesh>
}

type Collision = (Entity, Entity, Vector3<f32>);

impl PhysicsEngine {
    pub fn new() -> PhysicsEngine {
        PhysicsEngine {
            entity_count: 0,
            entity: Vec::new(),
            entity_properties: Vec::new()
        }
    }

    pub fn add_entity(&mut self, props: EntityProperties) -> Entity {
        let e = self.entity_count;
        self.entity_count += 1;
        self.entity_properties.push(props);
        self.entity.push(EntityState {
            prev_position: Vector3::zeros(),
            position: Vector3::zeros(),
            scale: Vector3::from([1.0, 1.0, 1.0]),
            rotation: UnitQuaternion::identity(),
            velocity: Vector3::zeros(),
            mass: 0.0,
            vertices: Rc::new(Vec::new()),
            mesh: Vec::new() }); // TODO parameter?
        e
    }

    pub fn set_entity(&mut self, entity: Entity) -> &mut EntityState {
        &mut self.entity[entity]
    }

    pub fn get_entity(&self, entity: Entity) -> &EntityState {
        &self.entity[entity]
    }

    pub fn time_step(&mut self, duration: f32) {
        let forces = self.forces();
        self.euler(duration, forces);
        let collisions = self.collide();
        self.resolve(collisions);
    }

    // Returns the current force applied to every entity
    fn forces(&self) -> Vec<Vector3<f32>> {
        let mut forces = vec![Vector3::zeros(); self.entity_count];
        for e in 0..self.entity_count {
            let entity = &self.entity[e];
            let properties = &self.entity_properties[e];
            if !properties.immovable {
                // Calculate how e is affected by other entities
                for other in 0..self.entity_count {
                    let other_entity = &self.entity[other];
                    let other_props = &self.entity_properties[other];
                    if other_props.gravitational {
                        // Gravity of other pulls on e
                        forces[e] += self.gravity(other_entity.position, entity.position, other_entity.mass, entity.mass);
                    }
                }
            }
        }
        forces
    }

    fn euler(&mut self, duration: f32, forces: Vec<Vector3<f32>>) {
        // Compute changes in velocity and position
        let dv = self.entity.iter().zip(forces).map(|(e, f)| f / e.mass * duration).collect::<Vec<_>>();
        let dp = self.entity.iter().map(|e| e.velocity * duration).collect::<Vec<_>>();
        // Apply changes to entities
        for e in 0..self.entity_count {
            self.entity[e].prev_position = self.entity[e].position; // Remember prev position in case of future collision
            self.entity[e].position += dp[e];
            self.entity[e].velocity += dv[e];
        }
    }

    fn collide(&mut self) -> Vec<Collision> {
        let mut collisions: Vec<Collision> = Vec::new();
        for e in 0..self.entity_count {
            let entity = &self.entity[e];
            let properties = &self.entity_properties[e];
            if !properties.immovable {
                // Check for collisions with other entities
                for other in 0..self.entity_count {
                    let other_entity = &self.entity[other];
                    let other_props = &self.entity_properties[other];

                    let self_to_world =
                        Translation3::from(entity.position).to_homogeneous() *
                        entity.rotation.to_homogeneous() *
                        Scale3::from(entity.scale).to_homogeneous();
                    let other_to_world =
                        Translation3::from(other_entity.position).to_homogeneous() *
                        other_entity.rotation.to_homogeneous() *
                        Scale3::from(other_entity.scale).to_homogeneous();
                    fn apply(mat: Matrix4<f32>, point: Vector3<f32>) -> Vector3<f32> { // TODO this is duped
                        let p = Point3::from_slice(point.as_slice());
                        let new = mat.transform_point(&p);
                        new.coords
                    }
                    if other_props.collision {
                        // Other's geometry may collide with entity; test meshes
                        let mut triangle_collisions = Vec::new();
                        for this_mesh in entity.mesh.iter() {
                            for other_mesh in other_entity.mesh.iter() {
                                let min1 = apply(self_to_world, this_mesh.bounding_min);
                                let max1 = apply(self_to_world, this_mesh.bounding_max);
                                let (min1, max1) = min1.inf_sup(&max1); // TODO what if bounding box is rotated??
                                let min2 = apply(other_to_world, other_mesh.bounding_min);
                                let max2 = apply(other_to_world, other_mesh.bounding_max);
                                let (min2, max2) = min2.inf_sup(&max2);
                                if PhysicsEngine::box_intersect(min1, max1, min2, max2) {
                                    // Meshes are proximal; test triangles
                                    triangle_collisions.append(&mut this_mesh.intersect_mesh(&other_mesh, self_to_world, other_to_world));
                                }
                            }
                        }
                        // Combine triangle collisions information into a single object collision by summing all intersected triangle normals
                        if !triangle_collisions.is_empty() {
                            let collision_normal = triangle_collisions.iter().sum::<Vector3<f32>>().normalize();
                            collisions.push((e, other, collision_normal));
                        }
                    }
                }
            }
        }
        collisions
    }

    fn resolve(&mut self, collisions: Vec<Collision>) {
        for (e, other, normal) in collisions {
            let entity = &mut self.entity[e];
            // Reflect vector off collision plane, with a restitution factor
            if entity.velocity.dot(&normal) < 0.0 { // If velocity is moving deeper into mesh
                entity.velocity -= 1.4 * normal * normal.dot(&entity.velocity); // Invert direction of normal component of velocity
            }
        }
    }

    fn box_intersect(min1: Vector3<f32>, max1: Vector3<f32>, min2: Vector3<f32>, max2: Vector3<f32>) -> bool {
        let [min1x, min1y, min1z]: [f32; 3] = min1.into();
        let [max1x, max1y, max1z]: [f32; 3] = max1.into();
        let [min2x, min2y, min2z]: [f32; 3] = min2.into();
        let [max2x, max2y, max2z]: [f32; 3] = max2.into();
        fn overlapping(min1: f32, max1: f32, min2: f32, max2: f32) -> bool {
            min1 <= max2 && min2 <= max1
        }
        overlapping(min1x, max1x, min2x, max2x) &&
        overlapping(min1y, max1y, min2y, max2y) &&
        overlapping(min1z, max1z, min2z, max2z)
    }

    fn in_box(min: Vector3<f32>, max: Vector3<f32>, point: Vector3<f32>) -> bool {
        point[0] >= min[0] && point[1] >= min[1] && point[2] >= min[2] &&
        point[0] <= max[0] && point[1] <= max[1] && point[2] <= max[2]
    }

    fn gravity(&self, source: Vector3<f32>, point: Vector3<f32>, m1: f32, m2: f32) -> Vector3<f32> {
        // Acceleration due to gravity is F = G * m1 * m2 / r^2
        let g = 0.0000006674_f64;
        let r2 = (point - source).norm_squared();
        let magnitude = (g * m1 as f64 * m2 as f64 / r2 as f64) as f32;
        let direction = (source - point).normalize();
        magnitude * direction
    }
}

pub struct Mesh {
    pub bounding_min: Vector3<f32>, // Contains all vertices in this portion of mesh
    pub bounding_max: Vector3<f32>,
    vertices: Rc<Vec<Vector3<f32>>>,
    indices: Vec<u16> // Vertices given in clockwise order
}

impl Mesh {
    pub fn new(vertices: Rc<Vec<Vector3<f32>>>, indices: Vec<u16>) -> Mesh {
        let (mut bounding_min, mut bounding_max) = (vertices[indices[0] as usize], vertices[indices[0] as usize]);
        for index in indices.iter() {
            let vertex = vertices[*index as usize];
            bounding_min = vertex.inf(&bounding_min);
            bounding_max = vertex.sup(&bounding_max);
        }
        Mesh { bounding_min, bounding_max, vertices, indices }
    }

    // Returns an iterator over all the triangles in this mesh, represented by triples of vertices
    pub fn triangles(&self) -> impl Iterator<Item = [&Vector3<f32>; 3]> {
        self.indices.chunks(3).map(|indices| {
            let [i, j, k]: [u16; 3] = indices.try_into().unwrap();
            [&self.vertices[i as usize], &self.vertices[j as usize], &self.vertices[k as usize]]
        })
    }

    // Intersects two meshes together, returning a list of the normals of the triangles of other which were intersected
    pub fn intersect_mesh(&self, other: &Mesh, self_to_world: Matrix4<f32>, other_to_world: Matrix4<f32>) -> Vec<Vector3<f32>> {
        // For every pair of a triangle in self and a triangle in other
        // TODO this is quadratic time! can it be improved?
        let mut intersections = Vec::new();
        for [a, b, c] in self.triangles() {
            for [d, e, f] in other.triangles() {
                fn apply(mat: Matrix4<f32>, point: &Vector3<f32>) -> Vector3<f32> {
                    let p = Point3::from_slice(point.as_slice());
                    let new = mat.transform_point(&p);
                    new.coords
                }
                let (a, b, c) = (
                    apply(self_to_world, a),
                    apply(self_to_world, b),
                    apply(self_to_world, c));
                let (d, e, f) = (
                    apply(other_to_world, d),
                    apply(other_to_world, e),
                    apply(other_to_world, f));
                let inter = Mesh::intersect_triangles(a, b, c, d, e, f);
                if let Some (_line_segment) = inter {
                    let tri_normal = (e - d).cross(&(f - d)).normalize();
                    if !tri_normal[0].is_nan() && !tri_normal[1].is_nan() && !tri_normal[2].is_nan() {
                        intersections.push(tri_normal); // triangle normal is in world space
                    }
                }
            }
        }
        intersections
    }

    pub fn intersect_triangles(a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>, d: Vector3<f32>, e: Vector3<f32>, f: Vector3<f32>)
            -> Option<(Vector3<f32>, Vector3<f32>)> {
        // First, intersect planes of self and other to find the line of intersection; ignore if parallel
        let normal_a = (b - a).cross(&(c - a));
        let normal_d = (e - d).cross(&(f - d));
        let direction = normal_a.cross(&normal_d).normalize();
        // Plane 1: [normal_a] . [point] = [normal_a . a]
        // Plane 2: [normal_d] . [point] = [normal_d . d]
        // Solve:   A . x = b
        // Fix z = 0 to find a single point on the line, TODO ASSUMING NOT VERTICAL
        let matrix_a = Matrix3::from_rows(&[normal_a.transpose(), normal_d.transpose(), Vector3::from([0.0, 0.0, 1.0]).transpose()]);
        let column_b = Vector3::from([normal_a.dot(&a), normal_d.dot(&d), 0.0]);
        let column_x = SVD::new_unordered(matrix_a, true, true).solve(&column_b, f32::EPSILON).expect("Failed to perform SVD");
        let point = Vector3::from([column_x[0], column_x[1], 0.0]);

        // Next, intersect this line with each edge of each triangle to find alpha and beta parametrized points of intersection
        if let Some ((alpha_a, beta_a)) = Mesh::intersect_line_with_tri(point, direction, a, b, c) {
            if let Some ((alpha_b, beta_b)) = Mesh::intersect_line_with_tri(point, direction, d, e, f) {
                let (mut alpha_a, mut beta_a, mut alpha_b, mut beta_b) = (alpha_a, beta_a, alpha_b, beta_b);
                if alpha_a > beta_a {
                    swap(&mut alpha_a, &mut beta_a);
                }
                if alpha_b > beta_b {
                    swap(&mut alpha_b, &mut beta_b);
                }
                // Finally, test if the two parametrized intervals along the line for each triangle overlap. If so, they intersect!
                if alpha_a <= beta_b && alpha_b <= beta_a {
                    let (alpha, beta) = (alpha_a.max(alpha_b), beta_a.min(beta_b));
                    let (start, finish) = (point + alpha * direction, point + beta * direction);
                    Some ((start, finish)) // Line segment at intersection of triangles
                } else { None }
            } else { None }
        } else { None }
    }

    // Returns parametrized points along line which intersect legs of triangle, if they exist
    fn intersect_line_with_tri(point: Vector3<f32>, direction: Vector3<f32>, a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> Option<(f32, f32)> {
        let mut vec = Vec::new();
        if let Some ((p_a, along_triangle_leg)) = Mesh::intersect_lines(point, direction, a, b - a) {
            if along_triangle_leg >= 0.0 && along_triangle_leg <= 1.0 {
                vec.push(p_a);
            }
        }
        if let Some ((p_b, along_triangle_leg)) = Mesh::intersect_lines(point, direction, b, c - b) {
            if along_triangle_leg >= 0.0 && along_triangle_leg <= 1.0 {
                vec.push(p_b);
            }
        }
        if let Some ((p_c, along_triangle_leg)) = Mesh::intersect_lines(point, direction, c, a - c) {
            if along_triangle_leg >= 0.0 && along_triangle_leg <= 1.0 {
                vec.push(p_c);
            }
        }
        if vec.len() == 0 {
            None
        } else if vec.len() == 1 {
            Some ((vec[0], vec[0]))
        } else {
            Some ((vec[0], vec[1]))
        }
    }

    // Returns the parametrizations along each given line of their intersection point, or None if the lines do not intersect
    pub fn intersect_lines(point_a: Vector3<f32>, direction_a: Vector3<f32>, point_b: Vector3<f32>, direction_b: Vector3<f32>) -> Option<(f32, f32)> {
        // point = alpha * direction_a + point_a
        // point = beta * direction_b + point_b
        // alpha * direction_a + point_a = beta * direction_b + point_b
        // alpha * direction_a - beta * direction_b = point_b - point_a
        // [direction_a ]T * [alpha] = [point_b - point_a]
        // [-direction_b]  * [beta ] = [point_b - point_a]
        // [0           ]  * [0    ] = [point_b - point_a]
        // A * x = b
        // 3x3 * 3x1 = 3x1

        let matrix_a = Matrix3x2::from_columns(&[direction_a, -1.0 * direction_b]);
        let column_b = point_b - point_a;
        let result = SVD::new_unordered(matrix_a, true, true).solve(&column_b, f32::EPSILON).ok().map(|s| (s[0], s[1]));
        result
    }

    // Returns list of triangle intersections within this mesh along the ray
    pub fn intersect_ray(&self, source: Vector3<f32>, dest: Vector3<f32>) -> Vec<(Vector3<f32>, Vector3<f32>, f32)> {
        let mut intersections = Vec::new();
        for i in (0..self.indices.len()).step_by(3) {
            let (a, b, c) = ( // For each triangle in mesh
                self.vertices[self.indices[i] as usize],
                self.vertices[self.indices[i + 1] as usize],
                self.vertices[self.indices[i + 2] as usize]);
            let normal = (b - a).cross(&(c - a)).normalize();

            // Intersect line of ray with plane of triangle
            // Ray: source + alpha * (dest - source) = point
            // Plane: (point - a) . normal = 0
            // alpha = ((a - source) . normal) / ((dest - source) . normal)
            let alpha = ((a - source).dot(&normal)) / ((dest - source).dot(&normal));
            if alpha >= 0.0 && alpha <= 1.0 { // alpha = 0 means source on plane; alpha = 1 means dest on plane; alpha c [0, 1] means ray intersects plane
                let point = source + alpha * (dest - source);
                if Mesh::projected_within_triangle(point, a, b, c) {
                    intersections.push((point, normal, (dest - point).norm()));
                }
            }
        }
        intersections
    }

    pub fn projected_within_triangle(projected: Vector3<f32>, a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> bool {
        // Check if projected point is within triangle using barycentric coordinates
        let (x1, x2, x3) = (a[0], b[0], c[0]);
        let (y1, y2, y3) = (a[1], b[1], c[1]);
        let (x, y) = (projected[0], projected[1]);
        let denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
        let alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
        let beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
        let gamma = 1.0 - alpha - beta;
        alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0
    }
}
