use super::Algorithm;

pub struct PolynomialRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub degree: usize,
    pub learning_rate: f64,
    pub iterations: usize,
}

impl PolynomialRegression {
    pub fn new(degree: usize, learning_rate:f64 ) -> Self {
        PolynomialRegression {
            weights: Vec::new(),
            bias: 0.0,
            degree,
            learning_rate,
            iterations: 1000,
        }
    }

    fn polynomial_features(&self, data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut transformed_data = Vec::new();
        for sample in data.iter() {
            let mut transformed_sample = Vec::new();
            transformed_sample.push(1.0); // Bias term
            for feature in sample.iter() {
                for power in 1..=self.degree {
                    transformed_sample.push(feature.powi(power as i32));
                }
            }
            transformed_data.push(transformed_sample);
        }
        transformed_data
    }

    fn fit(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        let transformed_data = self.polynomial_features(data);
        let n_samples = transformed_data.len();
        let n_features = transformed_data[0].len();

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        for _ in 0..self.iterations {
            let mut weight_gradients = vec![0.0; n_features];
            let mut bias_gradient = 0.0;

            for i in 0..n_samples {
                let prediction = self.predict_one(&transformed_data[i]);
                let error = prediction - labels[i];

                for j in 0..n_features {
                    weight_gradients[j] += error * transformed_data[i][j];
                }
                bias_gradient += error;
            }

            for j in 0..n_features {
                self.weights[j] -= self.learning_rate * weight_gradients[j] / n_samples as f64;
            }
            self.bias -= self.learning_rate * bias_gradient / n_samples as f64;
        }
    }

    fn predict_one(&self, data: &Vec<f64>) -> f64 {
        let mut result = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            result += weight * data[i];
        }
        result
    }
}

impl Algorithm for PolynomialRegression {
    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        self.fit(data, labels);
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
        let transformed_data = self.polynomial_features(data);
        transformed_data.iter().map(|sample| self.predict_one(sample)).collect()
    }
}
