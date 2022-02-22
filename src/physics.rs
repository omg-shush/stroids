use nalgebra::Vector3;

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
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub mass: f32,
    pub mesh: Vec<Mesh>
}

type Collision = (Entity, Entity, Vector3<f32>, Vector3<f32>);

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
        self.entity.push(EntityState { position: Vector3::zeros(), velocity: Vector3::zeros(), mass: 0.0, mesh: Vec::new() }); // TODO parameter?
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

    fn resolve(&mut self, collisions: Vec<Collision>) {
        for (e, other, projected, normal) in collisions {
            let other_position = self.entity[other].position;
            let entity = &mut self.entity[e];
            entity.position = projected + other_position; // TODO for now, move back to surface of triangle
            // Reflect vector off triangle's plane, with a restitution factor
            entity.velocity -= 2.0 * (0.4 * normal * normal.dot(&entity.velocity));
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
                    if other_props.collision {
                        // Other's geometry may collide with entity
                        let mut under_triangle = Vec::new();
                        for mesh in other_entity.mesh.iter() {
                            // Put entity position into other's model space
                            // TODO needs to account for scale/rotation!
                            let position = entity.position - other_entity.position;
                            // First, check if entity collides with mesh's bounding box
                            // TODO don't just check center of entity
                            if PhysicsEngine::in_box(mesh.bounding_min, mesh.bounding_max, position) {
                                // Next, check each triangle in the mesh
                                for i in (0..mesh.indices.len()).step_by(3) {
                                    let (a, b, c) = (
                                            mesh.vertices[mesh.indices[i] as usize],
                                            mesh.vertices[mesh.indices[i + 1] as usize],
                                            mesh.vertices[mesh.indices[i + 2] as usize]);
                                    let normal = (b - a).cross(&(c - a)).normalize();
                                    // Check if point is behind triangle
                                    let dist_from_plane = (position - a).dot(&normal);
                                    if dist_from_plane <= 0.0 && dist_from_plane >= -0.03 {
                                        // Check if projected point is within triangle using barycentric coordinates
                                        let projected = position - (dist_from_plane * normal);
                                        if Mesh::projected_within_triangle(projected, a, b, c) {
                                            // Within but underneath triangle; potential collision
                                            under_triangle.push((projected, normal, dist_from_plane));
                                        }
                                    }
                                }
                            }
                        }

                        if !under_triangle.is_empty() {
                            // Verify potential collisions by casting a ray from the triangle to the entity.
                            // Check triangle closest to entity to determine if entity inside mesh or outside
                            let mut closest = (Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), f32::INFINITY);
                            for (projected, normal, dist) in under_triangle {
                                let source = projected;
                                let dest = source + normal * dist;
                                let to_entity = dest - source;
                                if dist.abs() < closest.3 { // Initialize closest if not already
                                    closest = (projected, normal, to_entity, dist);
                                }
                                // Get bounding box of ray and check which meshes may intersect
                                let (min, max) = source.inf_sup(&dest);
                                for mesh in other_entity.mesh.iter() {
                                    if PhysicsEngine::box_intersect(min, max, mesh.bounding_min, mesh.bounding_max) {
                                        let mut ray_intersections = mesh.intersect_ray(source, dest);
                                        ray_intersections.push((closest.0, closest.1, closest.3));
                                        // Keep intersection closest to the entity
                                        let min_dist = ray_intersections.iter().map(|(_, _, dist)| dist.abs()).reduce(f32::min).unwrap();
                                        closest = ray_intersections.iter().find_map(|(proj, norm, dist)|
                                            if dist.abs() == min_dist { Some ((*proj, *norm, to_entity, *dist)) } else { None } ).unwrap();
                                    }
                                }
                            }
                            // Check if closest triangle intersection places entity inside or outside mesh, based on triangle normal
                            let to_entity = closest.2;
                            let normal = closest.1;
                            if normal.dot(&to_entity) < 0.0 { // Normal points away from entity; inside mesh
                                collisions.push((e, other, closest.0, closest.1));
                            }
                        }
                    }
                }
            }
        }
        collisions
    }

    fn box_intersect(min1: Vector3<f32>, max1: Vector3<f32>, min2: Vector3<f32>, max2: Vector3<f32>) -> bool {
        let [min1x, min1y, min1z]: [f32; 3] = min1.into();
        let [max1x, max1y, max1z]: [f32; 3] = max1.into();
        let [min2x, min2y, min2z]: [f32; 3] = min2.into();
        let [max2x, max2y, max2z]: [f32; 3] = max2.into();
        fn overlapping(min1: f32, max1: f32, min2: f32, max2: f32) -> bool {
            min1 <= max2 && min2 <= max1
        }
        overlapping(min1x, max1x, min2x, max2x) ||
        overlapping(min1y, max1y, min2y, max2y) ||
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

    fn euler(&mut self, duration: f32, forces: Vec<Vector3<f32>>) {
        // Compute changes in velocity and position
        let dv = self.entity.iter().zip(forces).map(|(e, f)| f / e.mass * duration).collect::<Vec<_>>();
        let dp = self.entity.iter().map(|e| e.velocity * duration).collect::<Vec<_>>();
        // Apply changes to entities
        for e in 0..self.entity_count {
            self.entity[e].position += dp[e];
            self.entity[e].velocity += dv[e];
        }
    }
}

pub struct Mesh {
    bounding_min: Vector3<f32>, // Contains all vertices
    bounding_max: Vector3<f32>,
    vertices: Vec<Vector3<f32>>,
    indices: Vec<u16> // Vertices given in clockwise order
}

impl Mesh {
    pub fn new(vertices: Vec<Vector3<f32>>, indices: Vec<u16>) -> Mesh {
        let min_x = vertices.iter().map(|v| v[0]).reduce(f32::min).unwrap();
        let min_y = vertices.iter().map(|v| v[1]).reduce(f32::min).unwrap();
        let min_z = vertices.iter().map(|v| v[2]).reduce(f32::min).unwrap();
        let max_x = vertices.iter().map(|v| v[0]).reduce(f32::max).unwrap();
        let max_y = vertices.iter().map(|v| v[1]).reduce(f32::max).unwrap();
        let max_z = vertices.iter().map(|v| v[2]).reduce(f32::max).unwrap();
        Mesh {
            bounding_min: Vector3::from([min_x, min_y, min_z]),
            bounding_max: Vector3::from([max_x, max_y, max_z]),
            vertices: vertices, indices: indices
        }
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
