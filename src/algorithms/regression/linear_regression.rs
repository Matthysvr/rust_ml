// src/regression/algorithms/linear_regression.rs
use super::Algorithm;

pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
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
