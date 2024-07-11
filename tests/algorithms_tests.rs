// tests/algorithms_tests.rs
extern crate RustML;

use RustML::algorithms::{Algorithm, LinearRegression};

#[test]
fn test_linear_regression() {
    let model = LinearRegression::new();
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let labels = vec![1.0, 2.0];
    model.train(&data, &labels);
    let result = model.predict(&data);
    // assertions
}
