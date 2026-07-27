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
[problems with the Newtonian method]

Let's examine this simple quadratic function

{{< media src="generated_images/simple_quadratic.png" alt="quadratic function" >}}

The blue lines show the true function, first derivative, and second derivative respectively, which we will assume are unknown; we only know the values at the points we evaluate. The orange line shows that the minimum of the true function is the root of the first derivative

Our goal is to find the minimum of the function, shown with the orange line. Our first point is $x = 3.5$ at the red dot. We learn that the derivative at this point is positive, meaning the function minimum must be to left, at a lower value of $x$. Under a first-order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce `x` a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (remember we're pretending like we can't see the blue line). In that case, the first derivative must be linear, and its slope equals the second derivative. Then we just solve a simple $y = mx + b$ equation to find where the derivative is zero, and we have know the function minimum[^1]

[^1]: The minimum must be at a point with zero derivative. If the derivative at some point isn't zero, then it can't be the minimum because there is some direction in whcih the function continues to decrease

More formally, lets approximate the true function with a a second-order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series):

{{< media src="generated_images/second_taylor.png" alt="taylor series" >}}

Our goal is to find the point where the derivative equals zero, so

{{< media src="generated_images/step_calculation.png" alt="formula for step" >}}

Then the minimum of $f$ is at $x_{\min} = x + s$, where $s$ is our step size of $\frac{f'(x)}{f''(x)}$. 

Newtonion optimization uses the curvature of the function to estimate where the minimum *ought* to be, assuming the function is quadratic. Even if it's not a perfect bowl, newtonion optimization can still find the minimum quickly. Let's see what that looks like in practice.

Here's visual representation of how newtonion optimization finds the minimum of a function compared to the more common gradient descent. The function here is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

{{< media src="media/videos/lmbfgs_explainer/1080p60/GradientVsNewtonian.mp4" caption="$f(x) = (1-x)^2 + 50(y-x^2)^2$" >}}

Using gradient descent to find the minimum requires walking down the canyon walls - initially moving *away* from the minimum - before tracing a path along the valley floor. In this demonstration, Gradient descent takes 12,000 steps to reach the minimum, while newtonion optimization gets there in only 5 steps.

Despite its apparent superiority, newtonion optimization has a couple drawbacks, which is why its less commonly used than gradient descent. The first is that it's sensitive to the curvature of the function you're trying to minimize; if the function is not well approximated by a quadratic curve, then newtonion optimization can give suboptimal results.

{{< media src="media/videos/lmbfgs_explainer/1080p60/SinusoidalValley.mp4" caption="$f(x) = \sin(x) + \sin(y)$" >}}

Here, gradient descent is able to find the minimum but Newtonian descent quickly gets stuck in a saddle point.

