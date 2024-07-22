// src/algorithms/mod.rs
pub trait Algorithm {
    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>);
    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64>;
}

pub mod regression;