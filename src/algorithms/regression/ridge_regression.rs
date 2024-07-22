use super::Algorithm;

pub struct RidgeRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub iterations: usize,
    pub alpha: f64,
}

impl RidgeRegression {
    pub fn new(alpha: f64, iterations: usize) -> Self {
        RidgeRegression {
            weights: Vec::new(),
            bias: 0.0,
            iterations,
            alpha,
        }
    }

    fn solve_linear_system(&self, a: Vec<Vec<f64>>, b: Vec<f64>) -> Vec<f64> {
        let n = a.len();
        let mut a = a;
        let mut b = b;
    
        for i in 0..n {
            let pivot = a[i][i];
            for j in i..n {
                a[i][j] /= pivot;
            }
            b[i] /= pivot;
    
            for k in 0..n {
                if k != i {
                    let factor = a[k][i];
                    for j in i..n {
                        a[k][j] -= factor * a[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }
        }
    
        b
    }

    fn fit(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        let n_samples = data.len();
        let n_features = data[0].len();
        self.weights = vec![0.0; n_features];

        let alpha_matrix: Vec<Vec<f64>> = (0..n_features)
            .map(|i| (0..n_features).map(|j| if i == j { self.alpha } else { 0.0 }).collect())
            .collect();

        let mut xtx: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n_features];
        let mut xty: Vec<f64> = vec![0.0; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                xty[j] += data[i][j] * labels[i];
                for k in 0..n_features {
                    xtx[j][k] += data[i][j] * data[i][k];
                }
            }
        }

        for i in 0..n_features {
            for j in 0..n_features {
                xtx[i][j] += alpha_matrix[i][j];
            }
        }

        self.weights = self.solve_linear_system(xtx, xty);
    }

    fn predict_one(&self, data: &Vec<f64>) -> f64 {
        data.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum::<f64>() + self.bias
    }
}

impl Algorithm for RidgeRegression {
    fn train(&mut self, data: &Vec<Vec<f64>>, labels: &Vec<f64>) {
        self.fit(data, labels);
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
        data.iter().map(|d| self.predict_one(d)).collect()
    }
}
