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

Let's examine this simple quadratic function, along with its first and second derivatives

{{< media src="generated_images/simple_quadratic.png" alt="quadratic function" >}}

Pretend we don't know the true shape of $f(x)$, we only know the value of it and its first and second derivatives at the point we've evaluated - the red dot. Our goal is to find the minimum of the function, shown with the orange line. Our first point is $x = 3.5$ at the red dot. We learn that the derivative at this point is positive, meaning the function minimum must be to left, at a lower value of $x$. Under a first-order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce $x$ a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (still pretending like we can't see the blue lines). In that case, the first derivative must be linear, and its slope equals the second derivative. Then we just solve a simple $y = mx + b$ equation to find where the derivative is zero, and we know the function minimum[^1]. Indeed we can see that the minimum of $f(x)$ is at the root of $f\prime$

[^1]: The minimum must be at a point with zero derivative. If the derivative at some point isn't zero, then it can't be the minimum because there is some direction in whcih the function continues to decrease

Newtonion optimization uses the curvature of the function to estimate where the minimum *ought* to be, assuming the function is quadratic. Even if it's not a perfect bowl, newtonion optimization can still find the minimum quickly. Let's see what that looks like in practice.

Here's visual representation of how newtonion optimization finds the minimum of a function compared to the more common gradient descent. The function here is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

{{< media src="media/videos/lmbfgs_explainer/1080p60/GradientVsNewtonian.mp4" caption="$f(x) = (1-x)^2 + 50(y-x^2)^2$" >}}

Using gradient descent to find the minimum requires walking down the canyon walls - initially moving *away* from the minimum - before tracing a path along the valley floor. In this demonstration, Gradient descent takes 12,000 steps to reach the minimum, while newtonion optimization gets there in only 5 steps.

Despite its apparent superiority, newtonion optimization has a couple drawbacks, which is why its less commonly used than gradient descent. The first is that it's sensitive to the curvature of the function you're trying to minimize; if the function is not well approximated by a quadratic curve, then newtonion optimization can give suboptimal results.

{{< media src="media/videos/lmbfgs_explainer/1080p60/SinusoidalValley.mp4" caption="$f(x) = \sin(x) + \sin(y)$" >}}

Here, gradient descent is able to find the minimum but Newtonian optimization quickly gets stuck in a saddle point. This function is incredibly poorly approximated by a quadratic, so this is a worst-case scenario. 

Newtonion optimization has one additional drawback, which will be present no matter how close the true function is to a quadratic and motivates the creation of BFGS. To see it, lets walk through the math.

In order to derive the algorithm for Newtonian optimization, lets start by approximating the true function with a a second-order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series):

{{< media src="generated_images/second_taylor.png" alt="taylor series" >}}

We said above that the minimum of $f(x+s)$ must be at a root of $f\prime$, so to find our step size, set the derivative of $f$ equal to $0$

{{< media src="generated_images/step_calculation.png" alt="formula for step" >}}

Then the minimum of $f$ is at $x_{\min} = x + s$, where $s$ is our step size of $-\frac{f\prime(x)}{f\prime \prime(x)}$. 

Lets generalize this to functions of multiple variables. $f\prime(x)$ becomes the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) $∇f$, a vector of partial first derivatives. $f\prime \prime(x)$ becomes the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) $B$, a matrix of partial second derivatives. We now have

{{< media src="generated_images/multivar_step_calculation_p1.png" alt="formula for step in multiple variables" >}}

Now that our second derivative term is a matrix instead of a scalar, we can no longer divide by it; we have to invert it.

{{< media src="generated_images/multivar_step_calculation_p2.png" alt="formula for step in multiple variables" >}}

And this brings us to the major problem with Newtonion optimization - that pesky $B^{-1}$. Gradient descent is only concerned with the Jacobian $∇f$ to determine each step, which scales lineraly with the number of inputs. But the Hessian $B$ grows quadratically with the number of inputs[^2]. Finding the Hessian for a function of one million variables (not very large by modern machine learning standards) would require calculating five-hundred-trillion unique partial second derivatives every step. The Hessian *then* needs to be inverted on each step, which is $O(n^{2.37})$ in the best case[^3].

[^2]: The Hessian [must be symmetric](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives), meaning the Hessian for a function of $n$ inputs has $\frac{(n-1)^2}{2} + n$ unique elements, instead of $n^2$

[^3]: and thats $O(n^{2.37})$ with respect to the number of elements in the matrix. A five-hundred-trillion element matrix takes between $5.32e27$ and $1.25e35$ operations to invert. On a CPU running three billion operations per second, the sun would explode before you were even 1% of the way there

As amazing as Newtonion optimization is, it suffers from a terrible case of combinatorial explosion, and is impractical for all but the smalles problems

## Enter: Broyden, Fletcher, Goldfarb, and Shanno

We are now thouroughly convinced that incorporating second-derivative information into our optimization algorithm is awesome, but doing so naively is impractical. We must enter the world of [Quasi-Newtonion methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) which seek to follow the wisdom of Newtonion Optimization without fully calculating the Hessian on each optimization step.

