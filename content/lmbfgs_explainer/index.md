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

{{< media src="generated_images/simple_quadratic.png" alt="quadratic function" themed="true">}}

Pretend we don't know the true shape of $f(x)$; we have evaluated $f$ at the red dot ($x = 3.5$)and have calculated the value of its first and second derivatives at that point. Our goal is to find the minimum of the function, shown with the orange line. We learn that the derivative at this point is positive, meaning the function minimum must be to left, at a lower value of $x$. Under a first-order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce $x$ a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (still pretending like we can't see the blue lines). In that case, the first derivative must be linear, and its slope is the value of the second derivative. Then we solve a simple $y = mx + b$ equation to find where the first derivative is zero, and we know the function minimum. Indeed we can see that the minimum of $f(x)$ is at the root of $f'$


Newtonion optimization uses the curvature of the function to estimate where the minimum *ought* to be, assuming the function is quadratic. Even if it's not a perfect bowl, newtonion optimization can still find the minimum quickly. Let's see what that looks like in practice.

Here's visual representation of how newtonion optimization finds the minimum of a function compared to the more common gradient descent. The function here is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

{{< media src="media/videos/lmbfgs_explainer/1080p60/GradientVsNewtonian.mp4" caption="$f(x) = (1-x)^2 + 50(y-x^2)^2$" >}}

Using gradient descent to find the minimum requires walking down the canyon walls - initially moving *away* from the minimum - before tracing a path along the valley floor. In this demonstration, Gradient descent takes 12,000 steps to reach the minimum, while newtonion optimization gets there in only 4 steps.

Despite this incredible feat, pure Newtonion optimization has a couple drawbacks, which is why it's almost never used in practice. The first is its sensitivity to the curvature of the function to be to minimized. If the function is not well approximated by a quadratic curve, then Newtonion optimization can give suboptimal results.

{{< media src="media/videos/lmbfgs_explainer/1080p60/SinusoidalValley.mp4" caption="$f(x) = \sin(x) + \sin(y)$" >}}

Here, gradient descent is able to find the minimum but Newtonian optimization quickly gets stuck in a saddle point. This function is incredibly poorly approximated by a quadratic, so this is a worst-case scenario. 

Newtonion optimization has one additional drawback, which will be present no matter how close the true function is to a quadratic, and which and motivates the creation of BFGS. To see it, lets walk through the math.

In order to derive the algorithm for Newtonian optimization, we start by approximating the true function with a a second-order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series)[^2]:

<div class="math">

$f(x_0 + s) = f(x_0) + f'(x_0) * s + \frac{1}{2}f''(x_0) * s^2$

</div>

[^2]: If you're unfamiliar with Taylor Series, you may recognize this equation from Physics class as the formula for the position $p$ of an object at some time $t$: $p(t) = p(t_0) + v t + \frac{1}{2} at^2$


Where $x_0$ is a point at which we've evaluated $f$ and $s$ is a proposed step. We said above that the minimum of $f(x_0+s)$ must be at a root of $f'$, so we can find the step that will bring us to the minimum by setting the derivative of $f$ equal to $0$


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

As amazing as Newtonion optimization is, it suffers from a terrible case of combinatorial explosion, and is impractical for all but the smallest problems

## Enter: Broyden, Fletcher, Goldfarb, and Shanno

We are now thouroughly convinced that incorporating second-derivative information into our optimization algorithm is awesome in theory, but doing so naively is impractical. We must enter the world of [Quasi-Newtonion methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) which seek to follow the wisdom of Newtonion Optimization without fully calculating the Hessian on each optimization step. The BFGS algorithm maintains an approximation $H \approx B^{-1}$ which gets updated as we go, saving us from having to calculate and then invert the Hessian for each and every step.

The BFGS algorithm works as follows. Starting with an initial estimate of $H = I$, a starting $x$ chosen arbitrarily, and let $k$ be the current iteration of the algorithm, then

<div clas = "math" id="bfgs-algorithm">

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

Which brings us to the point of this entire article. We will now derive the BFGS update formula, proofing that this algorithm does work as intended.

We start from the acknowledgement that $H_k$ is an imperfect approximation of the true inverse Hessian of $f$[^5]. The inverted Hessian *ought* to explain the change in gradient observed between positions $x_k$ and $x_{k+1}$ satisfing an inverted [Secant Equation](https://en.wikipedia.org/wiki/Secant_method)

[^5]: And unless $f$ was in fact quadratic, it doesn't even have a *single* true Hessian, but that's besides the point

<div class = "math">

$Hy = s$

</div>

But since $H_k$ is an imperfect approximation, it presumably does not...

<div class = "math">

$H_k y_k \neq s_k$

</div>

So our goal is to find a new matrix $H_{k+1}$ that *does* satisfy the equation, explaining the change in slope we just observed

<div class = "math">

$H_{k+1} y_k = s_k$

</div>

This forms a simple [system of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations) that ought to be familiar to most folks who've studied linear algebra [^6]. We run into a wrinkle, however: our system is horribly underspecified. We have a system of $n$ equations, but $  \frac{n(n-1)}{2}$ free variables[^7]. For $n > 2$ there are an infinite number of possible new $H$s that could satisfy our secant equation[^8]. 

[^6]: if it's not, see [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) for an explanation of how systems of equations can be viewed as matrix algebra, and vice versa

[^7]: $H$ is the inverse of $B$, and $B$ must be symmetric per footnote 3, so $H$ must be symmetric

[^8]: And if $n<=2$ we might as well use another algorithm

Of those infinite $H$s, it is hopefully uncontroversial that we want the one that is closest to $H_k$. After all, our estimate of the inverted Hessian builds up curvature information as we iterate, and we'd like to preserve as much of that information as possible. The candidate $H$ we choose for $H_{k+1}$ shall be the matrix that changes *as little as possible* from $H_k$ while still satisfying our criterea.

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


We want to stay as close to $H_k$ as possible, so our goal is to find some $H$ which minimizes $\lVert H - H_k \lVert _W$ subject to $Hy_k = s_k$[^9].

[^9]: And is symmetric. If $H$ isn't symmetric then it's not a proper approximation of the inverse Hessian

We are now ready to begin our derivation of the update algorithm[^10]

[^10]: Had we picked a measure of closeness other than a weighted Frobenius norm, we wouldn't be working with BFGS but with [DFP](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula), [SR1](https://en.wikipedia.org/wiki/Symmetric_rank-one), etc. There are a variety of quasi-newton methods out there, and this choice of "measure of closeness" is one of the primary differentiators

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

Now lets talk about that weight matrix $W$. We're never going to actually construct $\hat{H} = W^{\frac{1}{2}}HW^{\frac{1}{2}}$, so the choice of weight matrix is purely algebraic. Let's choose as our weight matrix $G$, the[^11] average Hessian of $f$

[^11]: (theoretical)

<div class="math">

$$
\begin{array}{c}
W=G \\ \text{such that} \\ y_k = Gs_k
\end{array}
$$

</div>

(Note that $G$ is paired with $s$ where $H$ was paired with $y$. $G$ is the theoretical true Hessian, not the inverse Hessian like $H$)

Using this relationship between $y_k$, $s_k$, and our chosen weight matrix, we redefine $\hat{y}$

<div class="math" id="y_hat-definitions">

$$
\def\arraystretch{1.5}
\begin{array}{ccl}
\hat{y} & = & W^{-\frac{1}{2}}y_k \\
\hat{y} & = & G^{-\frac{1}{2}}y_k \\ 
\hat{y} & = & G^{-\frac{1}{2}}Gs_k \\
\hat{y} & = & G^{\frac{1}{2}}s_k \\
\end{array}
$$

</div>

We then substitue the value of $G$ in for $W$ in our definition of $\hat{s}$ and we see...

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{ccl}
\hat{s} &=& W^{\frac{1}{2}}s_k \\
\hat{s} &=& G^{\frac{1}{2}}s_k\\
\hat{s} &=& \hat{y}
\end{array}
$$

</div>

Both $\hat{y}$ and $\hat{s}$ equal $G^{\frac{1}{2}}s_k$. Under our change of variable, the secant condition which our candidate $H$ must satisfy is

<div class="math">

$$
\hat{H}\hat{y} = \hat{y}
$$

</div>

We get one additional property for free: since $\hat{H}$ must be symmetric, that means the orthogonal compliment to $\hat{y}$ must be closed. For any $w$ orthogonal to $\hat{y}$

<div class="math">

$$
\hat{y}^{\top}\hat{H}w = w^{\top}\hat{H}\hat{y} = w^{\top}\hat{y} = 0
$$

</div>

So if $w$ is orthogonal to $\hat{y}$, then $\hat{H}w$ must also be orthogonal to $\hat{y}$. Our desired $\hat{H}$ must send $\hat{y}$ to $\hat{y}$, and must not send any vector in the orthogonal compliment in the $\hat{y}$-direction. Set this fact aside for now; we'll use it later in step 3 to prove that what we do then is in fact optimal.

So we're looking for a matrix $\hat{H}$ which maps $\hat{y}$ to itself, still mindful of staying as close to $\hat{H_k}$ as possible (and always symmetric). Our method will be thus: modify $\hat{H}_k$'s in order to cancel its current action on $\hat{y}$, then construct and add in a matrix that maps $\hat{y}$ as we desire [^12].

[^12]: if this also seems plucked from the sky, bear with me. It will all work out




### Step 2: Cancel Action On $\hat{y}$ 

Let $Q$ be a matrix that projects onto the subspace spanned by $\hat{y}$. In other words, for any vector $x$, the result of $Qx$ will be the portion of $x$ parallel to $\hat{y}$

{{< media src="generated_images/project_onto_y_p1.png" alt="Projecting the vector x onto the subspace spanned by y" themed="true">}}

Then lets define $x$ in terms of the portional parallel to $\hat{y}$, $Qx$, and the remaining part, which we'll call $z$

<div class="math">

$$
x = Qx + z
$$

</div>

$Q$ takes the general form

<div class="math">

$$
Q = \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
$$

</div>

if we solve for $z$, that portion of $x$ orthagonal to $\hat{y}$, we get

<div class="math">

$$
\def\arraystrech{1.5}
\begin{array}{ll}
z &= x - Qx \\
&= (I - Q)x \\
\end{array}
$$

</div>

So for any matrix $Q$ which projects onto a subspace $L$, the matrix $I-Q$ will project onto the subspace orthogonal to $L$. Lets call this complimentary matrix $P$

<div class="math">

$$
P = I - Q = I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
$$

</div>


---- PUT ANOTHER GRAPH HERE ----

Since $P$ projects onto a subspace orthogonal to $Span\{\hat{y}\}$, $P\hat{y} = 0$. We now have the mechanism to cancel action on $\hat{y}$. Our matrix for part 2 - something *like* $\hat{H}_k$ that maps $\hat{y}$ to 0 -  will be

<div class="math">

$$
P\hat{H}_kP
$$

</div>


The right $P$ kills all action on $\hat{y}$ and maps it to 0. The left $P$ ensures that this matrix remains symmetric.

<div class="math">

$$
P\hat{H}_kP\hat{y} = P\hat{H}_k(Py) = P\hat{H}_k0 = 0
$$

</div>

### Step 3: Map $\hat{y}$ to $\hat{y}$

For the second part of our construction we need a matrix that that maps $\hat{y}$ to itself. The identity matrix $I$ is an obvious choice, but we also have $Q$, the matrix we just defined above which projects onto $span\{\hat{y}\}$. We want the one that minimizes the norm $\lVert\hat{H} - \hat{H_k}\rVert _F$, so let's investigate how each of these choices affect the Frobenius norm.

The difference between $Q$ and $I$'s respective effects on the norm is easiest to see if we conduct a [change-of-basis](https://en.wikipedia.org/wiki/Change_of_basis) to a new vector space. Let $J$ be our new vector space with orthonormal basis $\{j_1, j_2, \ldots, j_n\}$. We define $j_1 = \frac{\hat{y}}{\lVert\hat{y}\rVert}$, and $\{j_2, \cdots , j_n\}$ as orthonormal vectors spanning the remainder of $J$. Because $J$ is an orthogonal matrix - composed of orthognal vectors each with a norm of 1 - the change of basis does not affect the Frobenius norm 

<div class="math">

$$
\lVert JMJ \rVert _F = \lVert M \rVert _F
$$

</div>

So the construction with the lowest norm in our new basis will have the lowest norm outside of it as well. We can define our existing approximation of the inverse Hessian generally as

<div class="math">

$$
[\hat{H}_k]_J = 
\begin{bmatrix}
a & r^{\top} \\
r & C
\end{bmatrix}
$$

</div>

Where $a$ is a scalar, $r$ is a vector of length $n-1$, and $C$ is an $(n-1)\times(n-1)$ symmetrical matrix[^13]. Next lets look at $Q$ in our new basis. For some matrix $M$ which maps vector space $A$ back to itself, the formula to change its basis to vectors space $B$ is

[^13]: This representation actually has nothing to do with the change-of-basis. We can always write out a symmetrical matrix this way. We just haven't needed to before now

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{c}
\text{if} \\
M: x\mapsto y,\ \ \ \ x,y \in A \\
\text{then} \\
[M]_B = P_{B\leftarrow A}^{-1}MP_{B\leftarrow A} \\
\text{such that} \\
[M]_B:[x]_B\mapsto[y]_B,\ \ \ \ [x]_B,[y]_B \in B
\end{array}
$$

</div>

In other words we can construct a transformation in the new basis which is equivalent to $M$ in the old basis by sandwhiching $M$ between the change-of-basis matrix $P_{B\leftarrow A}$ and its inverse. Now, the change-of-basis matrix is constructed by taking the basis vectors of the source vector space and replacing them with their respective [coordinate vectors](https://en.wikipedia.org/wiki/Coordinate_vector) in the target vector space. Because we're starting in the standard basis, the change-of-basis matrix is simply the basis vectors of the target vector space. So

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{Q}]_J &= P_{J\leftarrow E}^{-1} \hat{Q} P_{J\leftarrow E} \\
&= J^{-1}\hat{Q}J \\
&= J^{\top}\hat{Q}J & \text{because } J \text{ is orthogonal} \\
&= J^{\top}\frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}J \\
&= \frac{1}{\lVert\hat{y}\rVert^2}J^{\top}\hat{y}\hat{y}^{\top}J \\
&= \frac{1}{\lVert\hat{y}\rVert^2}(J^{\top}\hat{y})(\hat{y}^{\top}J)
\end{array}
$$

</div>

Now remember that we defined $J$ such that $j_1 = \frac{\hat{y}}{\lVert\hat{y}\rVert}$, and the remaining columns $\{j_2, \dotsc, j_n\}$ are orthogonal to the first column, which means they're orthogonal to $\hat{y}$.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
J^{\top}\hat{y} &= 
\begin{bmatrix}
j_{1-1} & j_{1-2} & \dotsc & j_{1-n} \\
j_{2-1} & j_{2-2} & \dotsc & j_{2-n}\\
\vdots & & \ddots  & \vdots\\
j_{n-1} & j_{n-2} & \dotsc & j_{n-n}
\end{bmatrix}
\begin{bmatrix}
\hat{y}_1 \\
\hat{y}_2 \\
\vdots \\
\hat{y}_n
\end{bmatrix}
= \begin{bmatrix}
\hat{y} \cdot j_1 \\
\hat{y} \cdot j_2 \\
\vdots \\
\hat{y} \cdot j_n
\end{bmatrix} 
= \begin{bmatrix}
\frac{\lVert \hat{y} \rVert ^2}{\lVert \hat{y} \rVert} \\
0 \\
\vdots \\
0
\end{bmatrix} \\
&= \begin{bmatrix}
\lVert \hat{y} \rVert \\
0 \\
\vdots \\
0
\end{bmatrix}
\end{array}
$$

</div>

The result of our other parenthetical follows the same algebra, but also falls out easily from the transpose

<div class="math">

$$
\hat{y}^{\top} J = (J^{\top}\hat{y})^{\top} = \begin{bmatrix} \lVert \hat{y} \rVert & 0 & \cdots & 0 \end{bmatrix}
$$

</div>

Then our definition of $\hat{Q}$ in the new basis is

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{Q}]_J &= \frac{1}{\lVert \hat{y} \rVert ^2} 
\begin{bmatrix} \lVert \hat{y} \rVert & 0 & \cdots & 0 \end{bmatrix}
\begin{bmatrix} \lVert \hat{y} \rVert \\ 0 \\ \vdots \\ 0 \end{bmatrix} \\
&= 
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
\end{array}
$$

</div>

Which should be obvious in hindsight. $Q$ was a matrix which projected onto $span\{\hat{y}\}$. Now that we're in a vector space where the first basis vector is in the direction of $\hat{y}$, $[Q]_J$ is a matrix which extracts the first element of any vector on which it acts, and zeroes out all other elements. 

Now for the definition of $\hat{P}$ in our new basis, which was the matrix which projected onto the orthogonal compliment of $span\{\hat{y}\}$[^14]

[^14]: From that statement alone one could guess the definition of $[\hat{P}]_J$, but we'll show it anyway

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{P}]_J &= J^{\top}(I - \hat{Q})J \\
&= J^{\top}IJ - J^{\top}\hat{Q}J \\
&= I - [\hat{Q}]_J \\
&= \begin{bmatrix}
0 & 0 \\
0 & I_{n-1}
\end{bmatrix}
\end{array}
$$

</div>

And therefore, our version of $\hat{H}_k$ with its action on $\hat{y}$ canceled is, in our new basis

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{P}\hat{H}_k\hat{P}]_J &= 
\begin{bmatrix}
0 & 0 \\
0 & I_{n-1}
\end{bmatrix}
\begin{bmatrix}
a & r^{\top} \\
r & C
\end{bmatrix}
\begin{bmatrix}
0 & 0 \\
0 & I_{n-1}
\end{bmatrix} \\
& = 
\begin{bmatrix}
0 & 0 \\
0 & I_{n-1}
\end{bmatrix}
\begin{bmatrix}
0 & r^{\top} \\
0 & C
\end{bmatrix} \\
&=
\begin{bmatrix}
0 & 0 \\
0 & C
\end{bmatrix}
\end{array}
$$

</div>

Now lets start putting all the pieces together and see how they affect the norm. Construction 1 using $\hat{Q}$ to send $\hat{y}$ to itself, and construction 2 using $I$ to do so

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{lcl}
[\hat{P}\hat{H}_k\hat{P} + \hat{Q}]_J &= 
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 \\
0 & C
\end{bmatrix} &=
\begin{bmatrix}
1 & 0 \\
0 & C
\end{bmatrix} \\
[\hat{P}\hat{H}_k\hat{P} + I]_J &= 
\begin{bmatrix}
1 & 0 \\
0 & I_{n-1}
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 \\
0 & C
\end{bmatrix} &=
\begin{bmatrix}
1 & 0 \\
0 & C+I_{n-1}
\end{bmatrix}
\end{array}
$$

</div>

And now the norms


<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
\lVert [\hat{P}\hat{H}_k\hat{P} + \hat{Q} - \hat{H}_k]_J \rVert _F &= \lVert
    \begin{bmatrix}
    1 & 0 \\
    0 & C
    \end{bmatrix}
    -
    \begin{bmatrix}
    a & r^{\top} \\
    r & C
    \end{bmatrix} \rVert _F \\
&= \lVert
    \begin{bmatrix}
    1-a & -r^{\top} \\
    -r & 0
    \end{bmatrix} \rVert _F\\
&= (1-a)^2 + \lVert r \rVert ^2 \\
\lVert [\hat{P}\hat{H}_k\hat{P} + I - \hat{H}_k]_J \rVert _F &= \lVert
    \begin{bmatrix}
    1 & 0 \\
    0 & C + I_{n-1}
    \end{bmatrix}
    -
    \begin{bmatrix}
    a & r^{\top} \\
    r & C
    \end{bmatrix} \rVert _F \\
&= \lVert
    \begin{bmatrix}
    1-a & -r^{\top} \\
    -r & I_{n-1}
    \end{bmatrix} \rVert _F\\
&= (1-a)^2 + \lVert r \rVert ^2 + (n-1)
\end{array}
$$

</div>

The latter of which is obviously bigger for $n > 1$. We can, in fact, prove that $\hat{Q}$ is the *best* possible matrix here, and that there does not exist any other matrix which maps $\hat{y}$ to itself and produces a smaller norm. 

In our changed basis, it must be the case that

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{H}_{k+1}]_J = 
\begin{bmatrix}
1 & 0 \\
0 & M
\end{bmatrix}
\end{array}
$$

</div>

where $M$ is an arbitrary $(n-1)\times(n-1)$ symmetric matrix. This must be true because $[\hat{H}_{k+1}]_J$ must map $\hat{y}$ to itself, and in our changed bases $\hat{y}$ only has a non-zero element in the first position. Therefore the first column of $[\hat{H}_{k+1}]_J$ must be 1 followed by 0s. And since the matrix is symmetric, the first row must be the same. The remaining elements, $M$, are free. So since $[\hat{H}_{k+1}]_J$ must take the form above, and $[\hat{H}_{k}]_J$ takes the general form $\begin{bmatrix}a & r^{\top} \\ r & C\end{bmatrix}$, then the candidate $[\hat{H}]_J$ which is minimally distant from $[\hat{H}_k]_J$ is the one where $M=C$, which is exactly our $[\hat{Q}]_J$ construction.

We have now proved not only that $Q$ is the optimal choice for the matrix that maps $\hat{y}$ to itself, but that the construction $\hat{P}\hat{H}_k\hat{P} + \hat{Q}$ is the optimal choice for $\hat{H}_k+1$, as it is the matrix closest to $\hat{H}_k$ which satisfies our secant condition

### Step 4: Undo The Change of Variable

Now that we've proved the optimality of our construction, the final step to get our true update algorithm is to reverse the change of variable. The tricky parts are done, and all that remains is some algebra to get our update formula.

Recall that $\hat{H}_{k+1} = W^{\frac{1}{2}}H_kW^{\frac{1}{2}}$, so $\hat{H}_{k+1} = W^{-\frac{1}{2}}\hat{H}_{k+1}W^{-\frac{1}{2}}$. Let $c = \hat{y}^{\top}\hat{y}$ for brevity.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} &= W^{-\frac{1}{2}}\begin{bmatrix}(I - \frac{\hat{y}\hat{y}^{\top}}{c})\hat{H}_k(I - \frac{\hat{y}\hat{y}^{\top}}{c}) + \frac{\hat{y}\hat{y}^{\top}}{c}\end{bmatrix}W^{-\frac{1}{2}} \\
\end{array}
$$

</div>


Starting with that last term

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{-\frac{1}{2}}\frac{\hat{y}\hat{y}^{\top}}{c}W^{-\frac{1}{2}} &= \frac{(W^{-\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{-\frac{1}{2}})}{c} \\
&= \frac{(W^{-\frac{1}{2}}W^{\frac{1}{2}}s_k)(W^{-\frac{1}{2}}\hat{y})^{\top}}{c} \\
&= \frac{s_ks_k^{\top}}{c}
\end{array}
$$

</div>

Now lets get $c$ out of there. Refer back to [figure 2.1.7](#y_hat-definitions) if you need a refresher. Those definitions will be quite important here.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
c &= \hat{y}^{\top}\hat{y} \\
&= (W^{\frac{1}{2}}s_k)^{\top}(W^{\frac{1}{2}}s_k) \\
&= s_k^{\top}W^{\frac{1}{2}}W^{\frac{1}{2}}s_k \\
&= s_k^{\top}Ws_k \\
&= s_k^{\top}y_k
\end{array}
$$

</div>

Let's call this final value $\frac{1}{\rho _k}$, so $\rho _k = \frac{1}{s_k^{\top}y_k}$. Then

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} &= W^{-\frac{1}{2}}[(I - \rho _k\hat{y}\hat{y}^{\top})\hat{H}_k(I - \rho _k\hat{y}\hat{y}^{\top})]W^{-\frac{1}{2}} + \rho _k s_ks_k^{\top} \\
&= W^{-\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{\frac{1}{2}}H_kW^{\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{-\frac{1}{2}} + \rho _k s_ks_k^{\top}
\end{array}
$$

</div>

Lets look at the right part of that first term

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{-\frac{1}{2}} &= W^{\frac{1}{2}}IW^{-\frac{1}{2}} - \rho _k (W^{\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{-\frac{1}{2}} )\\
&= I - \rho _k (W^{\frac{1}{2}}W^{\frac{1}{2}}s_k)(W^{-\frac{1}{2}}\hat{y})^{\top} \\
&= I -\rho _k (Ws_k)(W^{-\frac{1}{2}}W^{\frac{1}{2}}s_k) \\
&= I - \rho _k y_k s_k^{\top}
\end{array}
$$

</div>

Now the left part of the first term

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{-\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{\frac{1}{2}} &= W^{-\frac{1}{2}}IW^{\frac{1}{2}} - \rho _k (W^{-\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{\frac{1}{2}}) \\
&= I - \rho _k s_ky_k^{\top}
\end{array}
$$

</div>

Just like the term above, but we switched the sides of the $W^{\frac{1}{2}}$ and $W^{-\frac{1}{2}}$ so we switched which $\hat{y}$ turned into $s_k$ and which turned into $y_k$.

This means the update formula that gives a new estimated inverse Hessian, incorporating the new curvature information while staying as close to the old estimate as possible is

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} = (I - \rho _k s_ky_k^{\top})H_k(I - \rho _k y_ks_k^{\top}) + \rho _k s_ks_k^{\top}
\end{array}
$$

</div>

This is the same update formula as given in in the BFGS algorithm [above](#bfgs-algorithm), but unexpanded. Using the expanded formula, one can calculate $H_{k+1}$ without doing any full $(n \times n) \times (n \times n)$ matrix multiplications. Expanding this version into the one given above is left as an exercise for the reader.

## BFGS In Practice