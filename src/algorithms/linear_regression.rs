// src/algorithms/linear_regression.rs
use super::Algorithm;

pub struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            weights: Vec::new(),
            bias: 0.0,
        }
    }

    fn fit(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        let n_samples = data.len();
        let n_features = data[0].len();

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let learning_rate = 0.01;
        let epochs = 1000;

        for _ in 0..epochs {
            let mut weight_gradients = vec![0.0; n_features];
            let mut bias_gradient = 0.0;

            for i in 0..n_samples {
                let prediction = self.predict_one(&data[i]);
                let error = prediction - labels[i];

                for j in 0..n_features {
                    weight_gradients[j] += error * data[i][j];
                }
                bias_gradient += error;
            }

            for j in 0..n_features {
                self.weights[j] -= learning_rate * weight_gradients[j] / n_samples as f64;
            }
            self.bias -= learning_rate * bias_gradient / n_samples as f64;
        }
    }

    fn predict_one(&self, data: &Vec<f64>) -> f64 {
        let mut result = self.bias;
        for i in 0..data.len() {
            result += self.weights[i] * data[i];
        }
        result
    }
}

impl Algorithm for LinearRegression {
    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        self.fit(data, labels);
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
        data.iter().map(|sample| self.predict_one(sample)).collect()
    }
}

// src/algorithms/linear_regression.rs
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
