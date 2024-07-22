extern crate rust_ml;

use rust_ml::algorithms::Algorithm;
use rust_ml::algorithms::regression::PolynomialRegression;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train() {
        let mut model = PolynomialRegression::new(2, 0.001); // Specify the degree of the polynomial
        // create dummy data
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![7.0, 23.0, 47.0];
        model.train(&data, &labels);
        
        // The model should have learned some weights and bias
        assert!(!model.weights.is_empty(), "Weights should not be empty after training");
        assert!(!model.weights[0].is_nan(), "Weights should not be NaN after training");
        assert!(model.bias != 0.0, "Bias should not be zero after training");
        assert!(!model.bias.is_nan(), "Bias should not be NaN after training");

        // The weights and bias can be checked more rigorously if needed
        println!("Weights: {:?}", model.weights);
        println!("Bias: {:?}", model.bias);
    }

    #[test]
    fn test_predict() {
        let mut model = PolynomialRegression::new(2, 0.002); // Specify the degree of the polynomial

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![7.0, 23.0, 47.0];
        model.train(&data, &labels);
        
        let new_data = vec![vec![7.0, 8.0]];
        let result = model.predict(&new_data);
        
        assert_eq!(result.len(), 1, "Prediction result should have one element");
        
        let predicted_value = result[0];
        let expected_value = 79.0;
        let tolerance = 2.0;
        
        assert!(
            (predicted_value - expected_value).abs() < tolerance,
            "Predicted value ({predicted_value}) should be close to the expected value ({expected_value})"
        );

        println!("Predicted value: {:?}", predicted_value);
    }
}
