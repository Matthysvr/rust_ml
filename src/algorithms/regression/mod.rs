mod linear_regression;
pub use self::linear_regression::LinearRegression;

mod polynomial_regression;
pub use self::polynomial_regression::PolynomialRegression;

mod ridge_regression;
pub use self::ridge_regression::RidgeRegression;

pub use super::Algorithm;