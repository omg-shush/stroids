use std::vec::Vec;

struct ProductionInput {
    rate: i64,
    capacity: i64
}

struct ProductionNetwork {
    inputs: Vec<ProductionInput>
}