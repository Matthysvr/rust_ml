pub trait Algorithm {
    fn train(&self, data: &Vec<Vec<f64>>, labels: &Vec<f64>);
    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64>;
}

pub mod linear_regression;
pub mod logistic_regression;
pub mod decision_trees;
pub mod neural_networks;