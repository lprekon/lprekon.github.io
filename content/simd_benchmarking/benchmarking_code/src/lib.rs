/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `x`
pub fn basis_activation(i: usize, k: usize, x: f64, knots: &[f64]) -> f64 {
    if k == 0 {
        if knots[i] <= x && x < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
    let left_recursion = basis_activation(i, k - 1, x, knots);

    let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
    let right_recursion = basis_activation(i + 1, k - 1, x, knots);

    let result = left_coefficient * left_recursion + right_coefficient * right_recursion;
    return result;
}

/// Calculate the value of the B-spline at the given parameter `x`
pub fn b_spline(x: f64, control_points: &[f64], knots: &[f64], degree: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..control_points.len() {
        result += control_points[i] * basis_activation(i, degree, x, knots);
    }
    return result;
}