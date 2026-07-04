+++
date = '2026-06-27T15:00:00-05:00'
draft = true
title = 'LM-BFGS Explained'
summary = "An intuitive walkthrough and proof of the LM-BFGS optimization algorithm"  
+++


I assume you're passingly familiar with machine learning and comfortable enough with linear algebra to multiply matrices

machine learning is, at its core, the practice of iteratively minimizing a loss function. For some mathematical model and its set of weights, and some loss function measuring how wrong the model currently is, repeatedly adjust the weights until the loss function is as small as you can get it. For most machine learning endeavors, the algorithm by which one repeatedly tweaks their weights is some form of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent): pick a loss function that's differentiable, calculate its derivative with respect to each weight to determine how to tweak them, and repeat.

Gradient descent is a first order minimization algorithm - it relies on the first derivative of the function we're trying to minimize.

LM-BFGS (and its predecessor BFGS) are second order algorithms, which incorporate curvature information - the second derivative of the loss function - to accelerate the search.

## Primer: Newtonian optmization


Newtonion optimization rests on the idea that any function minima must have a derivative of zero. This should be obvious - if the derivative was not 0, then there exists a direction in which we can move and find a smaller function value.

[look at a quadratic. can find minimum in single step]
[target function need not be truly quadratic, as long as it is close]
[second order taylor expansion, followed by algebra]
[newtonion optimization works on any function reasonably approximated by a second-order polynomial. we iterate.]
[generalize to multiple variables]
[problems with the newtonian method]

Let's examine this simple quadratic function

![quadratic function](generated_images/simple_quadratic.png)

The blue lines show the true function, first derivative, and second derivative respectively, which we will assume are unknown; we only know the values at the points we evaluate.

Our goal is to find the minimum of the function, shown with the orange line. Our first point is `x = 3.5` at the red dot. We learn that the derivative at this point is positive, meaning the function minimum must be to left, at a lower value of `x`. Under a first-order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce `x` a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (remember we're pretending like we can't see the blue line). In that case, the first derivative must be linear, and its slope equals the second derivative. Then we just solve a simple `y = mx + b` equation to find where the derivative is zero, and we have know the function minimum[^1]

[^1]: The minimum must be at a point with zero derivative. If the derivative at some point isn't zero, then it can't be the minimum because there is some direction in whcih the function continues to decrease

More formally, lets approximate the true function with a a second-order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series):

![taylor series](generated_images/second_taylor.png)

Our goal is to find the point where the derivative equals zero, so

![formula for step](generated_images/step_calculation.png)

Then the minimum of `f` is at `x_min = x + s`, where `s` is our step size of `f'(x)/f''(x)`. 