use super::Algorithm;

pub struct LassoRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub iterations: usize,
    pub alpha: f64,
    pub learning_rate: f64,
}

impl LassoRegression {
    pub fn new(alpha: f64, iterations: usize, learning_rate: f64) -> Self {
        LassoRegression {
            weights: Vec::new(),
            bias: 0.0,
            iterations,
            alpha,
            learning_rate,
        }
    }

    fn fit(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        let n_samples = data.len();
        let n_features = data[0].len();
        self.weights = vec![0.0; n_features];

        for _ in 0..self.iterations {
            for i in 0..n_samples {
                let prediction = self.predict_one(&data[i]);
                let error = labels[i] - prediction;

                self.bias += self.learning_rate * error;

                for j in 0..n_features {
                    if self.weights[j] > 0.0 {
                        self.weights[j] -= self.learning_rate * (-data[i][j] * error + self.alpha);
                    } else {
                        self.weights[j] -= self.learning_rate * (-data[i][j] * error - self.alpha);
                    }
                }
            }
        }
    }

    fn predict_one(&self, data: &Vec<f64>) -> f64 {
        data.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum::<f64>() + self.bias
    }
}

impl Algorithm for LassoRegression {
    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        self.fit(data, labels);
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
        data.iter().map(|d| self.predict_one(d)).collect()
    }
}
