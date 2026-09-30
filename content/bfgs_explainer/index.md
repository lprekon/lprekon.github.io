+++
date = '2026-06-27T15:00:00-05:00'
draft = true
title = 'BFGS Explained'
summary = "An intuitive walkthrough and proof of the BFGS optimization algorithm"  
+++

Machine learning is, at its core, the practice of iteratively minimizing a loss function. For some mathematical model and its set of weights, and some loss function measuring how wrong the model currently is, one repeatedly adjusts the weights until the loss function is as small as one can get it. For most machine learning endeavors, the algorithm by which one repeatedly tweaks their weights is some form of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent): pick a loss function that's differentiable, calculate its derivative with respect to each weight to determine how to tweak them, and repeat.

Gradient descent is a first order minimization algorithm - it relies on the first derivative of the function we're trying to minimize.

There also exist second order algorithms, which incorporate curvature information - the second derivative of the loss function - to accelerate the search. The most straightforward second order algorithm is Newtonian Optimization. There are also a whole host of quasi-Newtonian algorithms which approximate Newtonian optimization, seeking similar performance at a fraction of the computational cost.

Today we're going to talk about a particular quasi-Newtonian algorithm called BFGS, named for the four mathematicians who invented it (independently at the same time).

While there are [plenty](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm#Algorithm) of [places](https://machinelearningmastery.com/bfgs-optimization-in-python/) on [the internet](https://www.cs.purdue.edu/homes/jhonorio/16spring-cs52000-quasinewton.pdf) that will tell you *about* the BFGS algorithm, none (in my opinion) do an adequate job explaining where it comes from. The algorithm involves some fairly arcane-looking linear algebra, and it is not immediately clear how that math achieves its stated purpose. Why and how does this algorithm actually *work*?

This article will explain the intuition behind second order optimization, present the BFGS algorithm, and then provide a derivation of and proof for the algorithm, showing that it is in fact optimal (given certain assumptions). A later article will continue the discussion to L-BFGS, a successor algorithm built on BFGS and which is more common in practice today.

Note: this article assumes familiarity with linear algebra concepts like matrix multiplication, transposition, and symmetry. Some other concepts - like projections and change-of-basis - are included with brief explanations

## Primer: Newtonian optimization

Newtonian optimization rests on two ideas: 
1) At any minimum of a function, the function's derivative at that point must be zero. This should be obvious - if the derivative were not 0, then there would exist a direction in which we can move and find a smaller function value.
2) The function to minimize is quadratic - i.e. it is twice-differentiable and has a constant second derivative[^1].


[^1]: Newtonian optimization can actually work on functions where this isn't strictly true, but it's a critical assumption for the algorithm

Let's examine this simple quadratic function, along with its first and second derivatives.

{{< media src="generated_images/simple_quadratic.png" alt="quadratic function" themed="true">}}

Pretend we don't know the true shape of $f(x)$; we have evaluated $f$ at the red dot ($x = 3.5$) and have calculated the value of its first and second derivatives at that point. Our goal is to find the minimum of the function, shown with the orange line. We learn that the derivative at this point is positive, meaning the function minimum must be to the left, at a lower value of $x$. Under a first order optimization algorithm like gradient descent, this is all the information we could glean; our next step would be to reduce $x$ a small amount and repeat. 

But let us now *assume* the true function we're trying to minimize is quadratic (still pretending like we can't see the blue lines). In that case, the first derivative must be linear, and its slope is the value of the second derivative. Then we solve a simple $y = mx + b$ equation to find where the first derivative is zero, and we know the function minimum. Indeed we can see that the minimum of $f(x)$ is at the root of $f'$.


Newtonian optimization uses the curvature of the function to estimate where the minimum *ought* to be, assuming the function is quadratic. Even if it's not a perfect bowl, we can still sometimes find the minimum quickly.

Here's a visual representation of how that works in practice, alongside the more common gradient descent. The function here is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).

{{< media src="media/videos/bfgs_explainer/1080p60/GradientVsNewtonian.mp4" caption="$f(x) = (1-x_0)^2 + 50(x_1-x_0^2)^2$" >}}

Using gradient descent to find the minimum requires walking down the canyon walls - initially moving *away* from the minimum - before tracing a path along the valley floor. In this demonstration, after 12,000 steps of gradient descent we're still only approaching the minimum, while Newtonian optimization gets to the minimum in only 5 steps.

Despite this incredible feat, pure Newtonian optimization has a couple of drawbacks, which is why it's almost never used in practice. The first is its sensitivity to the curvature of the function to be minimized. If the function is not well approximated by a quadratic curve, then we can get quite suboptimal results.

{{< media src="media/videos/bfgs_explainer/1080p60/SinusoidalValley.mp4" caption="$f(x) = \sin(x_0) + \sin(x_1)$" >}}

Here, gradient descent is able to find the minimum but Newtonian optimization quickly gets stuck in a saddle point. While Newtonian optimization can be *used* to find a function minimum, all the algorithm actually seeks is a nearby point where the gradient is zero. Here, the algorithm brings us to one such point, just not the one we would have preferred.

Newtonian optimization has one additional drawback, which will be present no matter how close the true function is to a quadratic, and which motivates the creation of BFGS. To see it, let's walk through the math.

In order to derive the algorithm for Newtonian optimization, we start by approximating the true function with a second order [Taylor Expansion](https://en.wikipedia.org/wiki/Taylor_series)[^2]:

<div class="math">

$f(x_0 + s) \approxeq f(x_0) + f'(x_0) * s + \frac{1}{2}f''(x_0) * s^2$

</div>

[^2]: If you're unfamiliar with Taylor Series, you may recognize this equation from Physics class as the formula for the position $p$ of an object at some time $t$: $p(t) = p(t_0) + v t + \frac{1}{2} at^2$.


Where $x_0$ is a point at which we've evaluated $f$ and $s$ is a proposed step. We said above that the minimum of $f(x_0+s)$ must be at a root of $f'$, so we can find the step that will bring us to the minimum by setting the derivative of $f$ equal to $0$.


<div class="math">

$$
\def\arraystretch{2}
\begin{array}{lr}
0 = \frac{d}{ds} [ f(x_0) + f'(x_0)*s + \frac{1}{2}f''(x_0)*s^2 ] \\
0 = f'(x_0) + f''(x_0) * s \\
s = - \frac{f'(x_0)}{f''(x_0)}, & f'' > 0\\
\end{array}
$$

</div>

So, starting from evaluating $f$ at some point $x_0$, the minimum of $f$ is at $x_{\min} = x_0 + s$, where $s$ is our step size of $-\frac{f'(x_0)}{f''(x_0)}$. In other words $\operatorname{argmin}_x f(x) = x_0 - \frac{f'(x_0)}{f''(x_0)}$ (assuming curvature greater than 0).

Let's generalize this to functions of multiple variables. $f'(x)$ becomes $\nabla f$, a vector of partial first derivatives. $f''(x)$ becomes the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) $B$, a matrix of partial second derivatives.

Starting with our definitions
<div class="math">

$$
\def\arraystretch{1}
\begin{aligned}
\nabla f =& \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}^\top \\[1.25em]
B =&
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix} 
\end{aligned}
$$

</div>

Then our Taylor expansion and subsequent root of the derivative become

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

And this brings us to the major problem with Newtonian optimization: that pesky $B^{-1}$. Gradient descent determines each step with only $\nabla f$, which scales linearly with the number of inputs. But the Hessian $B$ grows quadratically with the number of inputs[^3]. Finding the Hessian for a function of one million variables (not very large by modern machine learning standards) would require calculating five-hundred-billion unique partial second derivatives every step. The Hessian *then* needs to be inverted on each step, and inverting an $n \times n$ matrix is an $O(n^3)$ operation[^4].

[^3]: The Hessian [must be symmetric](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives), meaning the Hessian for a function of $n$ inputs has $\frac{n(n+1)}{2} $ unique elements, instead of $n^2$.

[^4]: A one-million-by-one-million square matrix takes $10\times10 ^{17}$ operations to invert. On a CPU running three billion operations per second, simply inverting the Hessian a single time would take about ten and a half years.

As amazing as Newtonian optimization is, the computation required grows cubically with the number of parameters. It is therefore impractical for all but the smallest problems.

## Enter: Broyden, Fletcher, Goldfarb, and Shanno

We are now thoroughly convinced that incorporating second-derivative information into our optimization algorithm is awesome in theory, but doing so naively is impractical. We must enter the world of [Quasi-Newtonian methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) which seek to follow the wisdom of Newtonian Optimization without fully calculating the Hessian on each optimization step. The BFGS algorithm maintains an approximation $H \approx B^{-1}$ which gets updated as we go, saving us from having to calculate and then invert the Hessian for each and every step.

The BFGS algorithm works as follows. With a starting $x$ chosen arbitrarily, calculate the first round of $f(x)$ and $\nabla f$. Let our starting estimate of $H = I$, and let $k$ be the current iteration of the algorithm. Then

<div class = "math" id="bfgs-algorithm">

$$
\def\arraystretch{1.5}
\begin{array}{cl}
1.&\text{Determine step direction } p_k = -H_k \nabla f_k \\
2.&\text{Perform a line search in direction } p_k \text { to find step } s_k = \alpha _k p_k \\
3.&\text{Update } x_{k+1} = x_k + s_k, \text{ calculate } f(x_{k+1}), \nabla f_{k+1} \\
4.&\text{Let } y_k = \nabla f_{k+1} - \nabla f_k \\
5.&\text{Update estimate of inverse Hessian }\\
& H_{k+1} = H_k + \frac{(s_{k}^\top y_k + y_{k}^\top H_k y_k)(s_k s_{k}^\top)}{(s_{k}^\top y_k)^2} - \frac{H_k y_k s_{k}^\top + s_k y_{k}^\top H_k}{s_{k}^\top y_k}
\end{array}
$$

</div>

After which we return to step one and repeat, now with a better understanding of the curvature of the function thanks to our updated $H$. 

If you think that update equation in step 5 fell from the sky and was bestowed upon us by aliens, you are not alone. 
Let's explain that line search in step two, and then we'll get to the point of this article - deriving the update algorithm and proving that it is optimal.

## A Line Search to Satisfy Wolfe Conditions

Step one of the algorithm gives us search direction $p_k$, but we don't necessarily want to step a full $\lvert p_k \rvert$. The only reason you'd be certain $s_k = p_k$ would be if A) $f$ was in fact purely quadratic and B) $H$ was the true inverse Hessian. Given that we want to apply this algorithm to functions besides quadratic functions, and that $H$ is merely an approximation of the inverse Hessian which we're building over time, neither A nor B hold. We need a way to more carefully pick how far in the direction $p_k$ we want to go. 

In order to figure out how far in the direction $p_k$ to travel between steps $k$ and $k+1$, we conduct a line search to satisfy [Wolfe conditions](https://en.wikipedia.org/wiki/Wolfe_conditions). All that means is we pick some starting $\alpha _k$ (usually $\alpha _k = 1$), evaluate $f(x_k + \alpha _k p_k)$ and $\nabla f(x_k + \alpha _k p_k)$, check them against certain conditions, and adjust $\alpha _k$ until our conditions are met. The two conditions we want satisfied are

<div class = "math">

$$
\def\arraystretch{1.5}
\begin{array}{cl}
1.&f(x_k + \alpha _k p_k) \le f(x_k) + c_1 \alpha _k p_k^{\top} \nabla f(x_k) \\
2.&-p_k^{\top} \nabla f(x_k + \alpha _k p_k) \le -c_2 p_k^{\top} \nabla f(x_k)
\end{array}
$$

</div>

Where $c_1$ and $c_2$ are arbitrary constants between $0$ and $1$, usually chosen as $10^{-4}$ and $0.9$ respectively. These two conditions place an upper and lower bound on $\alpha _k$.

The first condition stops us from picking an overly large $\alpha _k$ and seriously overshooting the minimum. Think of the right side of the inequality as a bar that starts at $f(x_k)$ when $\alpha = 0$. As our theoretical step size (and thus candidate choice of $\alpha _k$) increases, that bar drops. The further we step, the greater requirement we have for the decrease of $f$. Eventually, as we continue to increase $\alpha$, $f(x_k + \alpha p_k)$ will go past the function minimum and will pop up above the ever-decreasing horizontal bar, breaking the inequality and giving us our absolute maximum $\alpha_k$. This inequality doesn't completely stop us from overshooting the minimum, but it does put a cap on how far we can go. It can be a little hard to picture, so here's how it works in practice

{{< media src="media/videos/bfgs_explainer/1080p60/LineSearch.mp4" >}}

The second condition keeps us from picking an $\alpha _k$ so small that we go nowhere. In this inequality, we compare the gradient in our step direction at the proposed new $x$ with some portion of that same value at our current $x$. We insist that the slope of $f$ at our new spot moves away from $- \infty$[^5], and that it has moved at least an amount proportional to our old slope; lower values of $c_2$ require us to achieve greater movement away from $- \infty$.

[^5]: It's tempting to say "toward $0$" here, but that's technically inaccurate as it implies this condition would *stop* pushing us once the slope hits zero and wouldn't push the slope positive. The second condition is absolutely still satisfied if we overshoot the minimum and the slope becomes positive, and that's why we need the first condition.

Now, as to how one actually *finds* an appropriate value for $\alpha _k$, there are [a number of methods](https://en.wikipedia.org/wiki/Line_search). In fact, an astute reader may have noticed that this is actually an optimization problem itself. If you really wanted to, you could technically recurse and conduct a BFGS-based search for the minimum of the one dimensional function $g(\alpha) = f(x_k + \alpha p_k)$! This would be quite silly though, as you'd then need to do a line search in your new BFGS algorithm, recursing infinitely. 

In practice, the line search to satisfy Wolfe conditions is usually conducted using a bracketed line search. Essentially, we find a range that's guaranteed to contain a value for $\alpha$ that satisfies both conditions, and then conduct something like a binary search within that range until we find an $\alpha$ that's "good enough". We don't want to waste time finding the *optimal* $\alpha$. Once we find an $\alpha$ which satisfies our conditions, we're better off spending our compute resources finding the next direction of search in our overall algorithm than refining $\alpha$. Good enough is good enough.

The exact line search algorithm is a bit complicated and beyond the scope of this article. If you're interested, check out [PyTorch's implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/optim/lbfgs.py#L40).


## Deriving the Update Algorithm

Now for the headliner: how does one actually arrive at that long update algorithm? What is it doing that results in $H_{k+1}$ being a better approximation of the inverse Hessian than $H_k$?

We start from the acknowledgement that $H_k$ is an imperfect approximation of the true inverse Hessian of $f$[^6]. The inverse Hessian *ought* to explain the change in gradient observed between positions $x_k$ and $x_{k+1}$ satisfying an inverted [Secant Equation](https://en.wikipedia.org/wiki/Secant_method)

[^6]: And unless $f$ was in fact quadratic, it doesn't even have a *single* true Hessian, but that's beside the point

<div class = "math">

$Hy = s$

</div>

But since $H_k$ is an imperfect approximation, it presumably does not.

<div class = "math">

$H_k y_k \neq s_k$

</div>

So our goal is to find a new matrix $H_{k+1}$ that *does* satisfy the equation, explaining the change in slope we just observed

<div class = "math">

$H_{k+1} y_k = s_k$

</div>

This forms a simple [system of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations) that ought to be familiar to most folks who've studied linear algebra[^7]. We run into a wrinkle, however: our system is horribly underspecified. We have a system of $n$ equations, but $\frac{n(n+1)}{2}$ free variables[^8]. For $n >1$ there are an infinite number of possible new $H$s that could satisfy our secant equation[^9]. 

[^7]: if it's not, see [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) for an explanation of how systems of equations can be viewed as matrix algebra, and vice versa

[^8]: $H$ is the inverse of $B$, and $B$ must be symmetric per footnote 3, so $H$ must be symmetric

[^9]: And if $n = 1$ we might as well use a simpler algorithm

Of those infinite $H$s, it is hopefully uncontroversial that we want the one that is closest to $H_k$. After all, our estimate of the inverse Hessian builds up curvature information as we iterate, and we'd like to preserve as much of that information as possible. The candidate $H$ we want for $H_{k+1}$ is the matrix that changes *as little as possible* from $H_k$ while still satisfying our criteria.

In order to measure the difference, we might go with a simple Frobenius norm

<div class="math">

$$
\lVert M \rVert _F = \sqrt{\sum_i \sum_j m_{ij}^2}
$$

</div>

However the Frobenius norm is sensitive to the elements of our input $x$ being measured in different magnitudes; if say, $x_0$ was in meters but $x_1$ was in centimeters, a Frobenius norm might over-index on keeping the higher absolute value elements similar at the expense of other elements. So rather than a Frobenius norm, we'll use a weighted Frobenius norm.

Let $W$ be a matrix of weights. Then the weighted Frobenius norm is

<div class="math">

$$
\lVert M \rVert _W = \lVert W^{\frac{1}{2}} M W^{\frac{1}{2}} \rVert _F
$$

</div>


We want to stay as close to $H_k$ as possible, so our goal is to find some $H$ which minimizes $\lVert H - H_k \rVert _W$ subject to $Hy_k = s_k$[^10].

[^10]: And is symmetric. If $H$ isn't symmetric then it's not a proper approximation of the inverse Hessian.

We are now ready to begin our derivation of the update algorithm.



### Part 1: Change of Variable

Since we're going to weight our $H$s as part of measuring distance, let's talk about the weighted matrices

<div class="math">

$$
\begin{array}{c}
\hat{H} = W^{\frac{1}{2}} H W^{\frac{1}{2}} \\
\hat{H}_k = W^{\frac{1}{2}} H_k W^{\frac{1}{2}} \\
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
\lVert H - H_k \rVert _W = \lVert \hat{H} - \hat{H}_k \rVert _F
$$

</div>
 
And the secant condition we must satisfy is

<div class="math">

$$
\hat{H}\hat{y} = \hat{s}
$$

</div>

Now let's talk about that weight matrix $W$. We're never going to actually construct $\hat{H} = W^{\frac{1}{2}}HW^{\frac{1}{2}}$, so the choice of weight matrix is purely algebraic. Let's choose as our weight matrix $G$, the[^11] average Hessian of $f$

[^11]: (theoretical)

<div class="math">

$$
\begin{array}{c}
W=G \\ \text{such that} \\ y_k = Gs_k
\end{array}
$$

</div>

(Note that $G$ is paired with $s$ where $H$ was paired with $y$. $G$ is the hypothetical average Hessian, not the inverse Hessian like $H$)

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

We then substitute the value of $G$ in for $W$ in our definition of $\hat{s}$ and we see...

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

So we're looking for a matrix $\hat{H}$ which maps $\hat{y}$ to itself, still mindful of staying as close to $\hat{H}_k$ as possible (and always symmetric). Our method will be thus: modify $\hat{H}_k$ in order to cancel its current action on $\hat{y}$, then construct and add in a matrix that maps $\hat{y}$ as we desire [^12].

[^12]: If this also seems plucked from the sky, bear with me. It will all work out


### Part 2: Cancel Action On $\hat{y}$ 

Let $\hat{Q}$ be a matrix that projects onto the subspace spanned by $\hat{y}$. In other words, for any vector $x$, the result of $\hat{Q}x$ will be the portion of $x$ parallel to $\hat{y}$

{{< media src="generated_images/project_onto_y_p1.png" alt="Projecting the vector x onto the subspace spanned by y" themed="true">}}

Then let's define $x$ in terms of the portion parallel to $\hat{y}$ – $\hat{Q}x$ – and the remaining part, which we'll call $z$

<div class="math">

$$
x = \hat{Q}x + z
$$

</div>

$\hat{Q}$ takes the general form

<div class="math">

$$
\hat{Q} = \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
$$

</div>

If we solve for $z$, we get

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{ll}
z &= x - \hat{Q}x \\
&= (I - \hat{Q})x \\
\end{array}
$$

</div>

So for any matrix $\hat{Q}$ which projects onto a subspace $L$, the matrix $I-\hat{Q}$ will project onto the subspace orthogonal to $L$. Let's call this complementary matrix $\hat{P}$

<div class="math">

$$
\hat{P} = I - \hat{Q} = I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
$$

</div>


{{< media src="generated_images/project_onto_y_p2.png" alt="Projecting the vector x onto both y and its complement" themed="true">}}

Since $\hat{P}$ projects onto a subspace orthogonal to $\operatorname{Span}\{\hat{y}\}$, $\hat{P}\hat{y} = 0$. We now have the mechanism to cancel action on $\hat{y}$. The first part of our candidate $\hat{H}$ – something *like* $\hat{H}_k$ that maps $\hat{y}$ to 0 – will be

<div class="math">

$$
\hat{P}\hat{H}_k\hat{P}
$$

</div>


The right $\hat{P}$ kills all action on $\hat{y}$ and maps it to 0. The left $\hat{P}$ ensures that this matrix remains symmetric.

<div class="math">

$$
\hat{P}\hat{H}_k\hat{P}\hat{y} = \hat{P}\hat{H}_k(\hat{P}\hat{y}) = \hat{P}\hat{H}_k0 = 0
$$

</div>

### Part 3: Map $\hat{y}$ to $\hat{y}$

For the second part of our construction we need a matrix that maps $\hat{y}$ to itself. The identity matrix $I$ is an obvious choice, but we also have $\hat{Q}$, the matrix we just defined above which projects onto $\operatorname{Span}\{\hat{y}\}$. It turns out that $\hat{Q}$ is the better choice, producing an $\hat{H}_{k+1}$ which is closer to $\hat{H}_k$ than if we used $I$. We will prove this later, as part of proving that our full construction of $\hat{H}_{k+1}$ is optimal.


### Part 4: Undo The Change of Variable

Now that we've got all the pieces of our construction, we can put them together. 

In our changed variable, our update formula is

<div class="math">

$$
\hat{H}_{k+1} = \hat{P}\hat{H}_k\hat{P} + \hat{Q}
$$

</div>

The term on the left is a matrix close to $\hat{H}_k$, modified to map $\hat{y}$ to $0$, and the term on the right is a new matrix which maps $\hat{y}$ to itself, as required by our secant condition.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{ll}
\hat{H}_{k+1}\hat{y} &= ( \hat{P}\hat{H}_k\hat{P} + \hat{Q})\hat{y} \\
&= \hat{P}\hat{H}_k\hat{P}\hat{y} + \hat{Q}\hat{y}  \\
&= 0 + \hat{y} \\
&= \hat{y}
\end{array}
$$

</div>

Now all that's left is to undo the change of variable to get our true update formula. 

Recall that $\hat{H} = W^{\frac{1}{2}}HW^{\frac{1}{2}}$, so $H = W^{-\frac{1}{2}}\hat{H}W^{-\frac{1}{2}}$. 

Refer back to [figure 4.1.7](#y_hat-definitions) if you need a refresher. Those definitions will be quite important here.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} &= 
W^{-\frac{1}{2}}
\begin{bmatrix}
(I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}})\hat{H}_k(I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}) + \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
\end{bmatrix}
W^{-\frac{1}{2}} \\
&= W^{-\frac{1}{2}}
\begin{bmatrix}
(I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}})\hat{H}_k(I - \frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}})
\end{bmatrix}
W^{-\frac{1}{2}} + W^{-\frac{1}{2}}
\begin{bmatrix}\frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}
\end{bmatrix}
W^{-\frac{1}{2}}
\end{array}
$$

</div>

Let's start by changing back the variables in the numerator on the right part of the sum

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{-\frac{1}{2}}\frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}W^{-\frac{1}{2}} &= \frac{(W^{-\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{-\frac{1}{2}})}{\hat{y}^{\top}\hat{y}} \\
&= \frac{(W^{-\frac{1}{2}}W^{\frac{1}{2}}s_k)(W^{-\frac{1}{2}}\hat{y})^{\top}}{\hat{y}^{\top}\hat{y}} \\
&= \frac{s_k(W^{-\frac{1}{2}}W^{\frac{1}{2}}s_k)^{\top}}{\hat{y}^{\top}\hat{y}} \\
&= \frac{s_ks_k^{\top}}{\hat{y}^{\top}\hat{y}}
\end{array}
$$

</div>

Now we'll work the denominator

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
\hat{y}^{\top}\hat{y} &= (W^{\frac{1}{2}}s_k)^{\top}(W^{\frac{1}{2}}s_k) \\
&= s_k^{\top}W^{\frac{1}{2}}W^{\frac{1}{2}}s_k \\
&= s_k^{\top}Ws_k \\
&= s_k^{\top}y_k
\end{array}
$$

</div>

Let's call this final value $\frac{1}{\rho _k}$, so $\rho _k = \frac{1}{s_k^{\top}y_k}$. 

We'll substitute $\rho _k$ in, and change $\hat{H}_k$ back into $H_k$ 

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} &= W^{-\frac{1}{2}}[(I - \rho _k\hat{y}\hat{y}^{\top})\hat{H}_k(I - \rho _k\hat{y}\hat{y}^{\top})]W^{-\frac{1}{2}} + \rho _k s_ks_k^{\top} \\
&= W^{-\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{\frac{1}{2}}H_kW^{\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{-\frac{1}{2}} + \rho _k s_ks_k^{\top}
\end{array}
$$

</div>

Next up let's get the first piece of the sum back in terms of our original variable. Starting from the right-hand side of the sandwiched multiplication

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{-\frac{1}{2}} &= W^{\frac{1}{2}}IW^{-\frac{1}{2}} - \rho _k (W^{\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{-\frac{1}{2}} )\\
&= I - \rho _k (W^{\frac{1}{2}}W^{\frac{1}{2}}s_k)(W^{-\frac{1}{2}}\hat{y})^{\top} \\
&= I -\rho _k (Ws_k)(W^{-\frac{1}{2}}W^{\frac{1}{2}}s_k)^{\top} \\
&= I - \rho _k y_k s_k^{\top}
\end{array}
$$

</div>

Now we change back the variable on the left-hand side of the multiplication

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
W^{-\frac{1}{2}}(I - \rho _k\hat{y}\hat{y}^{\top})W^{\frac{1}{2}} &= W^{-\frac{1}{2}}IW^{\frac{1}{2}} - \rho _k (W^{-\frac{1}{2}}\hat{y})(\hat{y}^{\top}W^{\frac{1}{2}}) \\
&= I - \rho _k s_ky_k^{\top}
\end{array}
$$

</div>

This works out just like 4.4.7, but since the sides of the $W^{\frac{1}{2}}$ and $W^{-\frac{1}{2}}$ switched, we switched which $\hat{y}$ turned into $s_k$ and which $\hat{y}$ turned into $y_k$.

This means the update formula that gives a new estimated inverse Hessian, incorporating the new curvature information while staying as close to the old estimate as possible, is

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
H_{k+1} = (I - \rho _k s_ky_k^{\top})H_k(I - \rho _k y_ks_k^{\top}) + \rho _k s_ks_k^{\top}
\end{array}
$$

</div>

This is the same update formula as given in the BFGS algorithm above in step 5 of [fig 2.0.1](#bfgs-algorithm), but unexpanded. Expanding the formula allows one to calculate $H_{k+1}$ without doing any full $(n \times n) \times (n \times n)$ matrix multiplications, only multiplying $(n \times n)$ matrices with $n$-length vectors instead. Expanding this version into the one given above is left as an exercise for the reader.

## Proof

We have shown the full BFGS algorithm. We have shown how one may derive the BFGS update formula from first principles, hopefully providing some intuition about how and why it works. However, our derivation included a seemingly arbitrary choice of $\hat{Q}$ over $I$ as our $\hat{y}$-to-$\hat{y}$ projection matrix. Furthermore, just because the formula we derived *works* doesn't guarantee that it is the *best* formula[^13]. Conveniently, proving that $\hat{Q}$ is the correct choice of projection matrix leads us right to the proof that our full update formula results in the best possible $H_{k+1}$.

[^13]: Recall that here, "best" means the smallest weighted Frobenius norm of $H_{k+1} - H_k$, which is equivalent to the smallest un-weighted Frobenius norm of $\hat{H}_{k+1} - \hat{H}_k$.

We start by comparing how the choice of $\hat{Q}$ over $I$ affects the norm of the difference between $H_{k+1}$ and $H_k$. The difference between $\hat{Q}$'s and $I$'s respective effects on the norm is easiest to see if we conduct a [change-of-basis](https://en.wikipedia.org/wiki/Change_of_basis)[^14] to a new basis. Let $\mathcal J$ be our new basis with orthonormal basis $\{j_1, j_2, \ldots, j_n\}$. We define $j_1 = \frac{\hat{y}}{\lVert\hat{y}\rVert}$, and $\{j_2, \cdots , j_n\}$ as orthonormal vectors spanning the remainder of $\mathcal J$.

[^14]: In this article I'm using the notation from *Linear Algebra And Its Applications, 4th ed, by David C. Lay*, which is the inverse of Wikipedia's notation. In discussing changing the basis of a vector $x$ from an old basis to a new basis, Wikipedia uses the term "change-of-basis" matrix to refer to a matrix $A$ s.t $x_{old} = Ax_{new}$. Under the notation regime used in this article, the matrix described by Wikipedia would be called the change-of-coordinate matrix, and the change-of-basis matrix would be $B$ s.t $Bx_{old} = x_{new}$. Note that $B = A^{-1}$.

Since $\hat{H}_k$ is symmetric, we can define it in our new basis in the general form:

<div class="math">

$$
[\hat{H}_k]_{\mathcal J} = 
\begin{bmatrix}
a & r^{\top} \\
r & C
\end{bmatrix}
$$

</div>

Where $a$ is a scalar, $r$ is a vector of length $n-1$, and $C$ is an $(n-1)\times(n-1)$ symmetrical matrix[^15]. Next let's look at $\hat{Q}$ in our new basis. For any matrix $M$ which maps between vectors in some vector space $A$, the formula to change its basis to basis $B$ is as follows

[^15]: This representation actually has nothing to do with the change-of-basis. We can always write out a symmetrical matrix this way. We just haven't needed to before now

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{c}
\text{if} \\
M: x\mapsto y,\ \ \ \ x,y \in A \\
\text{then} \\
\begin{array}{cc}
[M]_B &= \underset{B\leftarrow A}{P}M\underset{A\leftarrow B}{P} \\
&= \underset{A \leftarrow B}{P}^{-1}M\underset{A \leftarrow B}{P}\\
\end{array} \\
\text{such that} \\
[M]_B:[x]_B\mapsto[y]_B,\ \ \ \ [x]_B,[y]_B \in B
\end{array}
$$

</div>

In other words if we want to apply the function represented by $M$ (which is defined in basis $A$) to vectors in basis $B$, simply transform those vectors into basis $A$, apply $M$, then transform them back; we construct $[M]_B$ by composing the functions which do so. In general, a change-of-basis matrix $\underset{B \leftarrow A}{P}$ is constructed by taking the basis vectors of the source basis - $A$ - and replacing them with their respective [coordinate vectors](https://en.wikipedia.org/wiki/Coordinate_vector) in the target basis. Let $J = [j_1, j_2, \cdots , j_n]$ be a matrix whose columns are the basis vectors of $\mathcal J$. Note that $J$ is an orthogonal matrix - each column has a norm of $1$ and is orthogonal to each other column. The columns of $J$ are already described in the standard basis $\mathcal E$, which means 

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
J &= \underset{\mathcal E \leftarrow \mathcal J}{P}\\
\underset{\mathcal J \leftarrow \mathcal E}{P} &= \underset{\mathcal E\leftarrow \mathcal J}{P}^{-1}\\
&= J^{-1} \\
&= J^{\top} & \text{because } J \text{ is orthogonal}
\end{array}
$$

</div>

Because $J$ is an orthogonal matrix the change of basis does not affect the Frobenius norm.

<div class="math">

$$
\lVert J^{-1}HJ \rVert _F = \lVert H \rVert _F
$$

</div>

If we prove our construction has the smallest norm in basis $\mathcal J$, then that proves it has the smallest norm in the standard basis $\mathcal E$

Now let us construct $[\hat{Q}]_{\mathcal J}$

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{ll}
[\hat{Q}]_{\mathcal J} &= \underset{\mathcal E\leftarrow \mathcal J}{P}^{-1} \hat{Q} \underset{\mathcal E\leftarrow \mathcal J}{P} \\
&= J^{-1}\hat{Q}J \\
&= J^{\top}\hat{Q}J \\
&= J^{\top}\frac{\hat{y}\hat{y}^{\top}}{\hat{y}^{\top}\hat{y}}J \\
&= \frac{1}{\lVert\hat{y}\rVert^2}J^{\top}\hat{y}\hat{y}^{\top}J \\
&= \frac{1}{\lVert\hat{y}\rVert^2}(J^{\top}\hat{y})(\hat{y}^{\top}J)
\end{array}
$$

</div>

Now remember that we defined $J$ as having a first column $j_1 = \frac{\hat{y}}{\lVert\hat{y}\rVert}$, and the remaining columns $\{j_2, \dotsc, j_n\}$ as orthogonal to the first column, which means they're orthogonal to $\hat{y}$.

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
J^{\top}\hat{y} &= 
\begin{bmatrix}
j_{1,1} & j_{1,2} & \dotsc & j_{1,n} \\
j_{2,1} & j_{2,2} & \dotsc & j_{2,n}\\
\vdots & \vdots & \ddots & \vdots\\
j_{n,1} & j_{n,2} & \dotsc & j_{n,n}
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

Then our definition of $\hat{Q}$ in basis $\mathcal J$ is

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{Q}]_{\mathcal J} &= \frac{1}{\lVert \hat{y} \rVert ^2} 
\begin{bmatrix} \lVert \hat{y} \rVert \\ 0 \\ \vdots \\ 0 \end{bmatrix} 
\begin{bmatrix} \lVert \hat{y} \rVert & 0 & \cdots & 0 \end{bmatrix} \\
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

Which should be obvious in hindsight. $\hat{Q}$ was a matrix which projected onto $\operatorname{Span}\{\hat{y}\}$. Now that we're in a basis where the first basis vector is in the direction of $\hat{y}$, $[\hat{Q}]_{\mathcal J}$ is a matrix which extracts the first element of any vector on which it acts, and zeroes out all other elements. 

Next up, the definition of $\hat{P}$ in our new basis, which was the matrix which projected onto the orthogonal complement of $\operatorname{Span}\{\hat{y}\}$[^16]

[^16]: From that statement alone one could guess the definition of $[\hat{P}]_{\mathcal J}$, but we'll show it anyway

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{P}]_{\mathcal J} &= J^{\top}(I - \hat{Q})J \\
&= J^{\top}IJ - J^{\top}\hat{Q}J \\
&= I - [\hat{Q}]_{\mathcal J} \\
&= \begin{bmatrix}
1 & 0 \\
0 & I_{n-1}
\end{bmatrix} -
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix} \\
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
[\hat{P}\hat{H}_k\hat{P}]_{\mathcal J} &= 
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

Now let's start putting all the pieces together and see how they affect the norm. As we do, we'll provide the pieces to prove our entire construction is optimal: construction 1 using $\hat{Q}$ to send $\hat{y}$ to itself, and construction 2 using $I$ to do so

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{lcl}
[\hat{P}\hat{H}_k\hat{P} + \hat{Q}]_{\mathcal J} &= 
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
[\hat{P}\hat{H}_k\hat{P} + I]_{\mathcal J} &= 
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

The norm of the difference between our first construction and $\hat{H}_{k}$ is


<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
\lVert [\hat{P}\hat{H}_k\hat{P} + \hat{Q} - \hat{H}_k]_{\mathcal J} \rVert _F &= \lVert
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
&= \sqrt{(1-a)^2 + 2\lVert r \rVert ^2}
\end{array}
$$

</div>

When we use the second construction, the norm is

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
\lVert [\hat{P}\hat{H}_k\hat{P} + I - \hat{H}_k]_{\mathcal J} \rVert _F &= \lVert
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
&= \sqrt{(1-a)^2 + 2\lVert r \rVert ^2 + (n-1)}
\end{array}
$$

</div>

The latter of which is obviously bigger for $n > 1$. We have therefore proved that using $\hat{Q}$ to construct $\hat{H}_{k+1}$ is better than using $I$. But might there exist some *other* matrix out there which would map $\hat{y}$ to itself and produce an even smaller norm than $\hat{Q}$? No, there mightn't. And in proving so, that $\hat{Q}$ is the best possible matrix here, we end up proving that $\hat{H}_{k+1} = \hat{Q} + \hat{P}\hat{H}_k\hat{P}$ is the best possible $\hat{H}_{k+1}$ given our choice of norm.

In our changed basis, it must be the case that

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{llr}
[\hat{H}_{k+1}]_{\mathcal J} = 
\begin{bmatrix}
1 & 0 \\
0 & M
\end{bmatrix}
\end{array}
$$

</div>

where $M$ is an arbitrary $(n-1)\times(n-1)$ symmetric matrix. This must be true because $[\hat{H}_{k+1}]_{\mathcal J}$ must map $\hat{y}$ to itself, and in our changed basis $\hat{y}$ only has a non-zero element in the first position. Therefore the first column of $[\hat{H}_{k+1}]_{\mathcal J}$ must be 1 followed by 0s. And since the matrix is symmetric, the first row must be the same. The remaining elements, $M$, are free. So since $[\hat{H}_{k+1}]_{\mathcal J}$ must take the form above, and $[\hat{H}_{k}]_{\mathcal J}$ takes the general form $\begin{bmatrix}a & r^{\top} \\ r & C\end{bmatrix}$, then the general form of the norm is


<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{ll}
\lVert [\hat{H}_{k+1}]_{\mathcal J} - [\hat{H}_k]_{\mathcal J} \rVert _F &= \lVert 
    \begin{bmatrix}
    1 & 0 \\
    0 & M
    \end{bmatrix} - \begin{bmatrix}
    a & r^{\top} \\
    r & C
    \end{bmatrix}
    \rVert _F \\
    &= \lVert \begin{bmatrix}
    1 - a & -r^{\top} \\
    -r & M - C
    \end{bmatrix} \rVert _F
\end{array}
$$

</div>

So the version of $[\hat{H}_{k+1}]_{\mathcal J}$ which is minimally distant from $[\hat{H}_k]_{\mathcal J}$ is the one where $M=C$, which is exactly our $[\hat{Q}]_{\mathcal J}$ construction.

We have now proved that $\hat{Q}$ is the optimal choice for the matrix that maps $\hat{y}$ to itself, and that the construction $\hat{P}\hat{H}_k\hat{P} + \hat{Q}$ is the optimal choice for $\hat{H}_{k+1}$, as it is the matrix closest to $\hat{H}_k$ – and thus the $H_{k+1}$ closest to $H_k$ – which satisfies our secant condition.

## BFGS In Practice

Let's take a look at how well BFGS handles the optimization scenarios we considered at the top of this article. Here's how BFGS handles our sin-based worst-case-for-Newtonian scenario

{{< media src="media/videos/bfgs_explainer/1080p60/BfgsSinusoidal.mp4" caption="BFGS $f(x) = \sin(x_0) + \sin(x_1)$" >}}

BFGS actually works much better on this problem than pure Newtonian optimization did! BFGS' approximations keep it from getting immediately stuck in the saddle point and give it a chance to find the true minimum. 

How does BFGS do at our Rosenbrock canyon?

{{< media src="media/videos/bfgs_explainer/1080p60/BfgsRosenbrok.mp4" caption="BFGS $f(x) = (1-x_0)^2 + 50(x_1-x_0^2)^2$" >}}

It certainly appears to take more steps than Newtonian optimization. One might wonder "if BFGS is learning the inverse Hessian, and updating that approximation with more and more information at each step, why does it crawl down the canyon, taking little steps each time? Why doesn't it eventually learn and take bigger steps?"

The answer is that this function has no single true Hessian, and thus no single inverse Hessian. As we walk along the canyon floor the approximate inverse Hessian is continuously being updated, but the true Hessian is also changing as we go, preventing us from ever learning "it" (because there is no single "it"), capping our step size.

Before one writes this exercise off and dismisses BFGS because it doesn't go any faster than gradient descent seemed to, it's worth noting that these animations are not perfectly to time-scale. When we watched gradient descent roll down the Rosenbrock canyon or roll into the sinusoid valley, we saw it move at constant speed; in reality, the first several steps are relatively large, and the vast majority of steps - the vast majority of iterations of that algorithm - occur right at the end. Take a look at this graph of training loss over amount of data trained on from [the LLaMA paper](https://arxiv.org/abs/2302.13971)

{{< media src="images/llama-training-curves.png" caption="arXiv:2302.13971, Figure 1">}}

Loss drops off quickly at the beginning, and then levels out as time goes on. Gradient descent is excellent at making quick, early steps toward the minimum, but is less effective the closer one gets. More advanced forms of gradient descent, like GD-with-momentum or Adam, address this issue and help gradient descent get closer to the minimum, but eventually they run into the same problem. 

The closer you get to a function minimum, the less the overall noise and general topology of that function matter. The more we zoom in around a function minimum (for just about any function), the more it starts to resemble a gently curving plain - a wide, flat bowl, very easily approximated by a quadratic curve. Which is exactly when BFGS works best! It is in these final approaches to the minimum that BFGS achieves [superlinear](https://en.wikipedia.org/wiki/Rate_of_convergence#Q-convergence) convergence: the ratio of the distance to the minimum on successive steps approaches zero as our steps approach infinity

<div class="math">

$$
\lim _{k\rightarrow \infty} \frac{\lVert x_{k+1} - x^*\rVert}{\lVert x_k - x^* \rVert} = 0
$$

</div>

Where $x^*$ is the true function minimum. Contrast this with simple gradient descent, which converges linearly in the best case[^17]

[^17]: True linear convergence of gradient descent requires [certain conditions](https://en.wikipedia.org/wiki/Gradient_descent#Theoretical_properties)

<div class="math">

$$
\def\arraystretch{1.5}
\begin{array}{lr}
\limsup _{k\rightarrow \infty} \frac{\lVert x_{k+1} - x^*\rVert}{\lVert x_k - x^*\rVert} = c ,& c \in (0, 1) \\
 \text{i.e} & \\
 \lVert x_{k+1} - x^* \rVert \leq c \lVert x_k - x^* \rVert ,& c \in (0, 1)
\end{array}
$$

</div>

The ratio of the distance to the minimum on successive steps has an upper bound of $c$ between $0$ and $1$, but always greater than zero; each step gets us only a fixed portion of the remaining distance, no matter how many steps we take.

Second order algorithms like BFGS are king when it comes to optimizations close to the function minimum. Hm... maybe one could start training with a cheap first order algorithm, and then switch to a more computationally expensive second order algorithm for fine tuning? But that - and a full explanation of BFGS's successor, L-BFGS - will be discussions for another day!