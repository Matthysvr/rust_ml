extern crate rust_ml;

use rust_ml::algorithms::Algorithm;
use rust_ml::algorithms::linear_regression::LinearRegression;

fn main() {
    // Create a new Linear Regression model
    let mut model = LinearRegression::new();
    
    // Create dummy data for training
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let labels = vec![5.0, 13.0, 21.0];
    
    // Train the model
    model.train(&data, &labels);
    
    // Display the trained weights and bias
    println!("Weights: {:?}", model.weights);
    println!("Bias: {:?}", model.bias);
    
    // Create new data for prediction
    let new_data = vec![vec![7.0, 8.0]];
    let result = model.predict(&new_data);
    
    // Display the prediction result
    println!("Predicted value: {:?}", result[0]);
    
    // Perform a check similar to the unit test
    let expected_value = 29.0; // Based on the training data pattern
    let tolerance = 1.0;
    
    assert!(
        (result[0] - expected_value).abs() < tolerance,
        "Predicted value should be close to the expected value"
    );
    
    println!("Prediction is within the expected range.");
}