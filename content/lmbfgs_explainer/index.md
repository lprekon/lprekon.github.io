+++
date = '2026-06-27T15:00:00-05:00'
draft = true
title = 'LM-BFGS Explained'
summary = "An intuitive walkthrough and proof of the LM-BFGS optimization algorithm"  
+++


I assume you're passingly familiar with machine learning and comfortable enough with linear algebra to multiply matrices

Machine learning is, at its core, the practice of iteratively minimizing a loss function. For some mathematical model and its set of weights, and some loss function measuring how wrong the model currently is, repeatedly adjust the weights until the loss function is as small as you can get it. For most machine learning endeavors, the algorithm by which one repeatedly tweaks their weights is some form of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent): pick a loss function that's differentiable, calculate its derivative with respect to each weight to determine how to tweak them, and repeat.

Gradient descent is a first order minimization algorithm - it relies on the first derivative of the function we're trying to minimize.

LM-BFGS (and its predecessor BFGS) are second order algorithms, which incorporate curvature information - the second derivative of the loss function - to accelerate the search.

## Primer: Newtonian optmization


Newtonion optimization rests on two ideas: 
1) At any minimum of a function, the function's derivative at that point must be zero. This should be obvious - if the derivative was not 0, then there exists a direction in which we can move and find a smaller function value.
2) The function to minimize is quadratic - i.e. it is twice-differentiable and has a constant second derivative[^1]


[^1]: Newtonion optimization can actually work on functions where this isn't strictly true, but its a load-bearing assumption for the algorithm

Let's examine this simple quadratic function, along with its first and second derivatives

{{< media src="generated_images/simple_quadratic.png" alt="quadratic function" >}}

Pretend we don't know the true shape of $f(x)$; we have evaluated $f$ at the red dot ($x = 3.5$)and have calculated the value of its first and second derivatives at that point. Our goal is to find the minimum of the function, shown with the orange line. We learn that the derivative at this point is positive, meaning the function minimum must be to left, at a lower value of $x$. Under a first-order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce $x$ a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (still pretending like we can't see the blue lines). In that case, the first derivative must be linear, and its slope is the value of the second derivative. Then we solve a simple $y = mx + b$ equation to find where the first derivative is zero, and we know the function minimum. Indeed we can see that the minimum of $f(x)$ is at the root of $f'$


Newtonion optimization uses the curvature of the function to estimate where the minimum *ought* to be, assuming the function is quadratic. Even if it's not a perfect bowl, newtonion optimization can still find the minimum quickly. Let's see what that looks like in practice.

Here's visual representation of how newtonion optimization finds the minimum of a function compared to the more common gradient descent. The function here is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

{{< media src="media/videos/lmbfgs_explainer/1080p60/GradientVsNewtonian.mp4" caption="$f(x) = (1-x)^2 + 50(y-x^2)^2$" >}}

Using gradient descent to find the minimum requires walking down the canyon walls - initially moving *away* from the minimum - before tracing a path along the valley floor. In this demonstration, Gradient descent takes 12,000 steps to reach the minimum, while newtonion optimization gets there in only 4 steps.

Despite this incredible feat, Newtonion optimization has a couple drawbacks, which is why its less commonly used than gradient descent. The first is its sensitivity to the curvature of the function to be to minimized. If the function is not well approximated by a quadratic curve, then Newtonion optimization can give suboptimal results.

{{< media src="media/videos/lmbfgs_explainer/1080p60/SinusoidalValley.mp4" caption="$f(x) = \sin(x) + \sin(y)$" >}}

Here, gradient descent is able to find the minimum but Newtonian optimization quickly gets stuck in a saddle point. This function is incredibly poorly approximated by a quadratic, so this is a worst-case scenario. 

Newtonion optimization has one additional drawback, which will be present no matter how close the true function is to a quadratic, and which and motivates the creation of BFGS. To see it, lets walk through the math.

In order to derive the algorithm for Newtonian optimization, we start by approximating the true function with a a second-order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series)[^2]:

<div class="math">

$f(x_0 + s) = f(x_0) + f'(x_0) * s + \frac{1}{2}f''(x_0) * s^2$

</div>

[^2]: If you're unfamiliar with Taylor Series, you may recognize this equation from Physics class as the formula for the position $p$ of an object at some time $t$: $p(t) = p(t_0) + v t + \frac{1}{2} at^2$


Where $x_0$ is a point at which we've evaluated $f$ and $s$ is a proposed step. We said above that the minimum of $f(x_0+s)$ must be at a root of $f'$, so we can find our step size by setting the derivative of $f$ equal to $0$


<div class="math">

$$
\def\arraystretch{2}
\begin{array}{l}
0 = \frac{d}{ds} [ f(x_0) + f'(x_0)*s + \frac{1}{2}f''(x_0)*s^2 ] \\
0 = f'(x_0) + f''(x_0) * s \\
s = - \frac{f'(x_0)}{f''(x_0)} \\
\end{array}
$$

</div>

Then, starting from evaluating $f$ at some point $x$, the minimum of $f$ is at $x_{\min} = x + s$, where $s$ is our step size of $-\frac{f'(x_0)}{f''(x_0)}$. In other words $argmin_x f(x) = x_0 + \frac{f'(x_0)}{f''(x_0)}$

Lets generalize this to functions of multiple variables. $f'(x)$ becomes the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) $∇f$, a vector of partial first derivatives. $f''(x)$ becomes the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) $B$, a matrix of partial second derivatives.

Starting with our definitions
<div class="math">

$$
\def\arraystretch{1}
\begin{aligned}
\nabla f =& \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}^\top \\[1.25em]
B =&
\begin{bmatrix}
\frac{\partial^2 f}{\partial^2 x_1} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial^2 x_n}
\end{bmatrix} 
\end{aligned}
$$

</div>

Then our taylor expansion and subsequent root of the derivative become

<div class="math">

$$
\def\arraystretch{2}
\begin{array}{rcl}
 f(x_0 + s) &=& f(x_0) + \nabla f^\top s + \frac{1}{2}(s^\top Bs) \\
 0 &=& \frac{d}{ds}[f(x_0) + \nabla f^\top s + \frac{1}{2}(s^\top Bs)]  \\
 0 &=& \nabla f + Bs \\
 -\nabla f &=& Bs \\
 -B^{-1}\nabla f &=& s
\end{array}
$$

</div>

And this brings us to the major problem with Newtonion optimization - that pesky $B^{-1}$. Gradient descent is only concerned with the Jacobian $∇f$ to determine each step, which scales lineraly with the number of inputs. But the Hessian $B$ grows quadratically with the number of inputs[^3]. Finding the Hessian for a function of one million variables (not very large by modern machine learning standards) would require calculating five-hundred-trillion unique partial second derivatives every step. The Hessian *then* needs to be inverted on each step, which is $O(n^{2.37})$ in the best case[^4].

[^3]: The Hessian [must be symmetric](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives), meaning the Hessian for a function of $n$ inputs has $\frac{(n-1)^2}{2} + n$ unique elements, instead of $n^2$

[^4]: and thats $O(n^{2.37})$ with respect to the number of elements in the matrix, not the number of inputs to the original function. A five-hundred-trillion element matrix takes between $5.32e27$ and $1.25e35$ operations to invert. On a CPU running three billion operations per second, the sun would explode before you were even 1% of the way there

As amazing as Newtonion optimization is, it suffers from a terrible case of combinatorial explosion, and is impractical for all but the smalles problems

## Enter: Broyden, Fletcher, Goldfarb, and Shanno

We are now thouroughly convinced that incorporating second-derivative information into our optimization algorithm is awesome, but doing so naively is impractical. We must enter the world of [Quasi-Newtonion methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) which seek to follow the wisdom of Newtonion Optimization without fully calculating the Hessian on each optimization step. The BFGS algorithm maintains an approximation $H = B^{-1}$ which gets updated at each step, saving us from having to invert the Hessian for each and every step.

The BFGS algorithm works as follows. Starting with an initial estimate of $H = I$, a starting $x$ chosen arbitrarily, and let $k$ be the current iteration of the algorithm, then

<div clas = "math">

$$
\def\arraystretch{1.5}
\begin{array}{cl}
1.&\text{Determine step direction } p_k = -H_k \nabla f_k \\
2.&\text{Perform a line search in direction } p_k \text { to find step } s_k = \alpha _k p_k \text{ (more on this later)} \\
3.&\text{Update } x_{k+1} = x_k + s_k, \text{ calculate } f(x_{k+1}), \nabla f_{k+1} \\
4.&\text{Let } y_k = \nabla f_{k+1} - \nabla f_k \\
5.&\text{Update estimate of inverse Hessian }\\
& H_{k+1} = H_k + \frac{(s_{k}^\top y_k + y_{k}^\top H_k y_k)(s_k s_{k}^\top)}{(s_{k}^\top y_k)^2} - \frac{H_k y_k s_{k}^\top + s_k y_{k}^\top H_k}{s_{k}^\top y_k}
\end{array}
$$

</div>

After which we return to step 1 and repeat, now with a better understanding of the curvature of the function thanks to our updated $H$. If you think that update equation in step 5 fell from the sky bestowed upon us by aliens, you are not alone. While there are [plenty](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm#Algorithm) of [places](https://machinelearningmastery.com/bfgs-optimization-in-python/) on [the internet](https://www.cs.purdue.edu/homes/jhonorio/16spring-cs52000-quasinewton.pdf) that will tell you *about* the BFGS algorithm, none (in my opinion) do an adquate job explaining where it comes from. 

We will now derive the BFGS update formula, prove that this works and/or provide some intution as to *why* this works.

We start from the acknowledgement that $H_k$ is an imperfect approximation of the true inverse Hessian of $f$[^5]. The inverted Hessian *ought* to explain the change in gradient observed between positions $x_k$ and $x_{k+1}$ satisfing the [Secant Equation](https://en.wikipedia.org/wiki/Secant_method)

[^5]: And unless $f$ was truly merely quadratic, it doesn't even have a *single* true Hessian, but that's besides the point

<div class = "math">

$Hy = s$

</div>

But since $H_k$ is an imperfect approximation, it presumably does not...

<div class = "math">

$H_k y_k \neq s_k$

</div>

So our goal is to find a new matrix $H_{k+1}$ that *does* satisfy the equation

<div class = "math">

$H_{k+1} y_k = s_k$

</div>

This forms a simple [system of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations) that ought to be familiar to most folks who've studied linear algebra [^6]. We run into a wrinkle though: our system is underspecified. We have a system of $n$ equations, but $  \frac{n(n-1)}{2}$ free variables[^7]. There are an infinite number of possible new $H$s that could satisfy our secant equation. 

[^6]: if it's not, see [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) for an explanation of how systems of equations can be viewed as matrix algebra, and vice versa

[^7]: $H$ is the inverse of $B$, and $B$ must be symmetric per footnote 3, so $H$ must be symmetric

Of those infinite $H$s, lets pick the one thats closest to $H_k$. Our estimate of the inverted Hessian builds up curvature information as we iterate, and we'd like to preserve as much of that information as possible. The candidate $H$ we choose for $H_{k+1}$ shall be the matrix that changes *as little as possible* from $H_k$ while still satisfying our criterea.

In order to measure the difference, we might go with a simple Frobenius norm

<div class="math">

$$
\lVert M \rVert _F = \sqrt{\sum_i \sum_j m_{ij}^2}
$$

</div>

However the Forbenius norm is sensitive to the elements of $x$ being measured in different magnitudes; If, say, $x_0$ was in meters but $x_1$ was in centimeters, a Frobenius norm might over-index on keeping the higher absolute value elements similar at the expense of other elements. So rather than a Frobenius norm, we'll use a weighted Frobenius norm.

Let $W$ be a matrix of weights. Then the weighted Frobenius norm is

<div class="math">

$$
\lVert M \rVert _W = \lVert W^{\frac{1}{2}} M W^{\frac{1}{2}} \rVert _F
$$

</div>


We want to stay as close to $H_k$ as possible, so our goal is to find $H$ which minimizes $\lVert H - H_k \lVert _W$ subject to $Hy_k = s$. 

We are now ready to begin our derivation of the update algorithm[^8]

[^8]: Had we picked a measure of closeness other than a weighted Frobenius norm, we wouldn't be working with BFGS but with [DFP](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula), [SR1](https://en.wikipedia.org/wiki/Symmetric_rank-one), etc. There are a variety of quasi-newton methods out there, and this choice of "measure of closeness" is one of the primary differentiators

### Step 1: Change of Variable

Since we're going to weight our $H$s as part of measuring distance, lets talk about the weighted matrices

<div class="math">

$$
\begin{array}{c}
\hat{H} = W^{\frac{1}{2}} H W^{\frac{1}{2}} \\
\hat{H_k} = W^{\frac{1}{2}} H_k W^{\frac{1}{2}} \\
\end{array}
$$

</div>

Then notice that 

<div class="math">

$$
\begin{array}{lc}
&Hy_k=s_k \\
\text{becomes}&\\
&(W^{-\frac{1}{2}}\hat{H}W^{-\frac{1}{2}})y_k = s_k
\end{array}
$$ 
 
</div>
 
when we back-substitute for $\hat{H}$. Left-multiplying both sides by $W^{\frac{1}{2}}$ gives us $\hat{H}(W^{-\frac{1}{2}}y_k) = (W^{\frac{1}{2}}s_k)$, so

<div class="math">

$$
\begin{array}{c}
\hat{y} = W^{-\frac{1}{2}}y_k \\
\hat{s} = W^{\frac{1}{2}}s_k
\end{array}
$$

</div>

Now our measure of distance is

<div class="math">

$$
\lVert H - H_k \rVert _W = \lVert \hat{H} - \hat{H_k} \rVert _F
$$

</div>
 
And the secant condition we must satisfy is

<div class="math">

$$
\hat{H}\hat{y} = \hat{s}
$$

</div>

Now lets talk about that weight matrix $W$. We're never going to actually construct $\hat{H} = W^{\frac{1}{2}}HW^{\frac{1}{2}}$, so the choice of weight matrix is purely algebraic. Let's choose as our weight matrix $G$, the[^9] average Hessian of $f$

<div class="math">

$$
\begin{array}{c}
W=G \\ \text{such that} \\ y_k = Gs_k
\end{array}
$$

</div>

(Note that $G$ is paired with $s$ where $H$ was paired with $y$. $G$ is the theoretical true Hessian, not the inverse Hessian like $H$)

[^9]: (theoretical)
 