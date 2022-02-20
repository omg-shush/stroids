use nalgebra::Vector3;

pub type Entity = usize;

pub struct PhysicsEngine {
    entity_count: usize,
    entity: Vec<EntityState>,
    entity_properties: Vec<EntityProperties>,
    mesh: Vec<(Mesh, Entity)>
}

pub struct EntityProperties {
    pub immovable: bool, // If this object should move due to simulation
    pub collision: bool, // If other objects should collide with this
    pub gravitational: bool // If other objects should experience attraction to this
}

#[derive(Debug)]
pub struct EntityState {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub mass: f32
}

impl PhysicsEngine {
    pub fn new() -> PhysicsEngine {
        PhysicsEngine {
            entity_count: 0,
            entity: Vec::new(),
            entity_properties: Vec::new(),
            mesh: Vec::new()
        }
    }

    pub fn add_entity(&mut self, props: EntityProperties) -> Entity {
        let e = self.entity_count;
        self.entity_count += 1;
        self.entity_properties.push(props);
        self.entity.push(EntityState { position: Vector3::zeros(), velocity: Vector3::zeros(), mass: 0.0 }); // TODO parameter?
        e
    }

    pub fn set_entity(&mut self, entity: Entity) -> &mut EntityState {
        &mut self.entity[entity]
    }

    pub fn get_entity(&self, entity: Entity) -> &EntityState {
        &self.entity[entity]
    }

    pub fn add_mesh(&mut self, entity: Entity, m: Mesh) {
        self.mesh.push((m, entity));
    }

    pub fn time_step(&mut self, duration: f32) {
        let forces = self.forces();
        self.euler(duration, forces);
        let collisions = self.collide();
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
                    if other_props.collision {
                        // Other's geometry may collide with entity
                    }
                }
            }
        }
        forces
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

    fn collide(&self) -> Vec<(Entity, Entity)> {
        Vec::new()
    }
}

pub struct Mesh {
    bounding_min: Vector3<f32>, // Contains all vertices
    bounding_max: Vector3<f32>,
    vertices: Vec<Vector3<f32>>,
    indices: Vec<usize> // Vertices given in clockwise order
}
