// tests/algorithms/linear_regression_test.rs
extern crate rust_ml;

use rust_ml::algorithms::Algorithm;
use rust_ml::algorithms::regression::LinearRegression;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train() {
        let mut model = LinearRegression::new();
        // create dummy data
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![3.0, 7.0, 11.0];
        model.train(&data, &labels);
        
        // The model should have learned some weights and bias
        assert!(!model.weights.is_empty(), "Weights should not be empty after training");
        assert!(model.bias != 0.0, "Bias should not be zero after training");

        // The weights and bias can be checked more rigorously if needed
        println!("Weights: {:?}", model.weights);
        println!("Bias: {:?}", model.bias);
    }

    #[test]
    fn test_predict() {
        let mut model = LinearRegression::new();
        // create dummy data
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![3.0, 7.0, 11.0];
        model.train(&data, &labels);
        
        // create new data for prediction
        let new_data = vec![vec![7.0, 8.0]];
        let result = model.predict(&new_data);
        
        // Check the prediction result
        assert_eq!(result.len(), 1, "Prediction result should have one element");
        
        // Since the exact value might vary due to training, check if it is reasonable
        let predicted_value = result[0];
        let expected_value = 15.0; // Based on the training data pattern
        let tolerance = 1.0;
        
        assert!(
            (predicted_value - expected_value).abs() < tolerance,
            "Predicted value should be close to the expected value"
        );

        println!("Predicted value: {:?}", predicted_value);
    }
}
