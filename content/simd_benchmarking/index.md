+++
date = '2025-02-12T19:00:59-05:00'
draft = true
title = 'Benchmarking Different Vectorization Strategies in Rust'
+++
# 

**Outline**
* What is vectorization/SIMD
* Different ways to vectorize in Rust
  * Compiler auto-vectorization
  * Rust’s portable SIMD crate
  * (Bonus) SIMBA
  * Platform specific
    * X86 vs ARM
* The algorithm we’ll vectorize
  * Sliding window method
  * Pyramid method
* What did we learn

{{< toc >}}

The first part of this blog post gives a brief overview of what SIMD operations and vectorization is and why you should care. After that I go over the different ways to get SIMD operations in to your Rust code, before finally diving into my different attempts to vectorize B-Spline calculations. If you’re familiar with writing SIMD Rust code and are just interested in seeing this particular example, skip to section 3. If you’re familiar with SIMD operations in general but not in Rust, skip to section 2. If you’re not familiar with SIMD and are ok with a brief overview before diving in, continue on. If you’re not familiar with SIMD and want a more thorough introduction before diving in here, I’ll direct you to McYoung’s delightful explainer (about a 45 minute read) https://mcyoung.xyz/2023/11/27/simd-base64/



## What is Vectorization/SIMD 
(Skip to section 2 if you’re already familiar with SIMD operations)	


Computers are really cool. They do computing. And they do it at speeds orders of magnitude faster than you or I can. In the time it takes us to calculate 2+2, your computer can figure out 2+2 a billion times over. But because we’re naturally greedy little monkeys, this still isn’t fast enough. Your CPU is incredibly fast. Incredibly fast. In fact, it’s about as fast as it can reasonably get [insert citation here]. There’s not a hard limit, but basic physics - power and heat dissipation (, plus quantum tunneling) means that making your CPU run any faster - say, calculating 2+2 five billion times over - is impractical. So, computer engineers have naturally built additional ways to increase computation speed, like multithreading

I hate writing I hate writing I hate writing

Think of your CPU like a motorcycle. Fast, efficient, adaptable; good at weaving between alley-ways and through traffic. If you were going on a scavenger hunt all around a city, it’s exactly the tool you would want. But if you’re carrying packages from Houston to Austin, a motorcycle ain’t the best tool. Sure, it’s fast, but even going felony-speeds that’s a 4 hour round-trip. Delivery half-dozen-packages would take a full 24 hours.

But if you already know all the packages are coming from one place and going to another, you’re not going to use a motorcycle - you’re better off using a truck. Sure, it takes a little longer to make the trip, but when it can deliver hundreds of packages at once, as opposed to the single package at a time the motorcycle can manage. Of course if we’re running from point-to-point around town, the truck might not be the best choice.

Back to computers, the motorcycle is your CPU, and the truck is your GPU. Your CPU can perform operations very quickly, but there is some overhead. For every operation it’s got to pull in the instruction and the data, perform the operation, possibly store the results, and then go back for more. If the next operation depends on the results of the current one, then it’s the best tool you’ve got. But, if you know all your packages are going to the same place - that is to say, if you have a lot of data, and you want to do the same operation on all of it - then you’re better off running on a GPU, the 18-wheeler to your CPU’s motorcycle.

But sometimes you don’t want to run your code on a GPU. Maybe you don’t have one, maybe you don’t feel like writing GPU code [insert link], or maybe you just don’t have enough data to make it worth it. If your CPU is the motorcycle delivering one package at a time, and your GPU is an 18-wheeler delivering hundreds, what do you do when you have a handful of packages? Then you turn to… a motorcycle with a side car! It turns out your CPU actually has some parallel processing capabilities like your GPU does (maybe, depending on how fancy it is [insert link]). These operations, called Single Instruction Multiple Data (SIMD) or sometimes vector operations, take multiple arguments in parallel and perform the same operation across them. For example, doing an element-wise add across two lists of numbers. In a regular CPU context, your processor would grab the first number from each list, add them together, write the result back to memory, then repeat on the next set of numbers. Using SIMD operations, the CPU can pull several numbers at a time from each list, add each chunk together, and write the entire chunk back at once [include graphics here]. We’re going to exercise this oft-unused circuitry to do some math in Rust, and compare and contrast different methods for doing so.

^^ maybe delete this bit and just direct people to McYoung


When we write a loop like this [show simple loop], that gets compiled down into assembly that looks like this [x86 intel syntax version] [arm version]

This code checks the loop condition, jumping over the loop body if the condition is false, and continuing into the loop body otherwise. At the end of the loop body, we jump back to the top of the loop and continue. About half of this loop is useful business logic, and about half - the branching and jumping - is what we’d consider overhead. If this loops runs a few times at the start of your program and never again, then it’s probably fine to leave it alone. But if this loop runs constantly and is at the core of your program, then it’s what we call a “hot” loop, and it might be worth it to try and improve performance. 

One way you could improve performance is by “unrolling” the loop, doing multiple steps per loop iteration [insert more pictures]. Loop unrolling increases the amount of useful work done per unit-overhead, or conversely, reduces the amount of overhead required to do a unit of useful work. We’re going to “vectorize” our loops, doing essentially the same thing, but by using SIMD operations instead of loop unrolling, we’ll get even greater benefit. It’ll look something like this [picture of loop with x86 vector operations]

The next section talks about different ways to get your compiled Rust code to include SIMD operations, and after that we’ll focus on the particular algorithm I optimized with SIMD instructions.

## Different Ways to Vectorize in Rust

For rest of this post, we’re going to be working in Rust. If you’re not a Rust developer, there will still be useful information here, but also, why aren’t you [insert link to why rust is great]? We’ll be looking at three different ways to get our Rust code compiled into SIMD operations: 1) The compiler’s auto-vectorization [link] 2) Rust’s portable SIMD [link] crate, and 3) CPU-specific intrinsics, for both 64 bit x86 and 64 bit ARM. Each method will have its own performance/portability/ease-of-use trade offs

### Rust's Auto-Vectorizer
One of the great things about Rust is that the compiler will automatically turn your boring old scalar instructions into shiny awesome vector instructions. Well, actually LLVM does the auto-vectorization, so any compiler build on top of LLVM - Rust, [insert others with links] - will get the same treatment. 

## The Algorithm We'll Vectorize
Ok, let’s talk about the actual algorithm we want to speed up with SIMD operations: we’ll be calculating the value of [B-Splines](https://en.wikipedia.org/wiki/B-spline). If you’ve worked in CAD or digital graphics products, you may have heard the term, and know it as “that tool that lets me draw curvy lines by moving points around, even though the line doesn’t go exactly through the points”. In general, [Splines](https://en.wikipedia.org/wiki/Spline_(mathematics)) are piece-wise polynomial functions known for their ability to trace out arbitrary curves, and B-Splines are a specific way to define and build splines. If you’re unfamiliar with the math behind B-splines, here’s a brief primer, so you can understand the code later

### A Brief Primer on B-Splines
A B-spline is a recursive piece-wise function defined using three values
1. A list of numbers called knots which define the intervals considered by the piece-wise function
2. A list of numbers called control points (or coefficients) which weight the different pieces of the function
3. A single number called the degree of the spline, which determines how many levels of recursion the function uses and, consequently, how smooth the resulting curve is

Let's look at an example:
<!-- <img src="generated_images/bspline_degree_0.png" alt="demo" class="img-responsive" max-width/> -->
![A degree-0 B-spline](generated_images/bspline_degree_0.png)

This is a dirt-simple degree-0 b-spline. The value of the spline at some value `x` is equal to the weighted sum of the constituent "basis functions" at `x` so long as `x` is within the range defined by the knots, and zero everywhere else. In the above example, the weights (or “control points”) are all 1 for simplicity. Let’s take a look at some of the basis functions for this degree-0 spline

![the 1st basis function for a degree-0 B-spline](generated_images/degree_0_basis_0.png)
![the 2nd basis function for a degree-0 B-spline](generated_images/degree_0_basis_1.png)
![the 3rd basis function for a degree-0 B-spline](generated_images/degree_0_basis_2.png)

Each degree 0 basis function is simply defined as `1` when `x` is between the `ith` and `i+1th` knot, and 0 everywhere else. Ok, so far, so boring. We’re just looking at some lines. Let’s start 

looking at higher degree B-splines to see how it comes together. Here’s general form of the basis function for degree 1 and higher

![The basis function formula for B-splines degree 1 and higher](generated_images/basis_formula.png)

That looks complicated, but we can break it down:
1. The `i'th` basis function for some degree `k` b-spline is equal to…
   1. The weighted combination of…
      1. The `i'th` basis function of the `k-1` degree b-spline 
      2. And…
      3. The `i+1'th` basis function of the `k-1` degree b-spline
      * (remember that the 0th degree basis functions are just 1 or 0 as defined above, so the recursion will end eventually)
   2. Where the weights are…
      1. Based on the “distance” between `x` and…
         1. The `i'th` knot, for the “left” basis function
         2. The `i+k'th` knot, for the “right” basis function
      2. Normalized by the length of the interval between the 
         1. `i+k'th` and `i'th` knot on the left
         2. `i+k+1'th` and `i+1'th` knot on the right

That’s a lot of math. Let’s look at in action. Here are the first three basis functions for a degree-1 B-spline

![the 1st basis function for a degree-1 B-spline](generated_images/degree_1_basis_0.png)
![the 2nd basis function for a degree-1 B-spline](generated_images/degree_1_basis_1.png)
![the 3rd basis function for a degree-1 B-spline](generated_images/degree_1_basis_2.png)

And for degree-2 basis functions:

![the 1st basis function for a degree-2 B-spline](generated_images/degree_2_basis_0.png)
![the 2nd basis function for a degree-2 B-spline](generated_images/degree_2_basis_1.png)
![the 3rd basis function for a degree-2 B-spline](generated_images/degree_2_basis_2.png)

All of these basis functions follow the formula defined above, where `k` equals the “degree” of the b-spline, and `i` is given the value 1, 2, or 3 for the first, second, and third images in each set, respectively.

Let’s move up to a degree-3 b-spline and put all the basis functions together

![A full degree-3 B-spline with control points all set to 1](generated_images/bspline_degree_3_full.png)

The colored lines are each one of our basis functions, and the black line is the full spline. At any point `x`, the value of `spline(x)` is the sum of the values of each basis function `B_i` evaluated at that point `x`. In the above example the control points are all set to 1. Let's see another example with different control points to see how splines are used to approximate different functions

![A full degree-3 B-spline with varying control points ](generated_images/bspline_degree_3_full_with_control_points.png)

Now we're cooking with gas! Here we see a B-spline in all it's glory. By manipulating the control points, we can "tug" the spline curve in one direction or another. 

Through the proper choice of knots, control points, and degree, we can use B-Splines to construct arbitrary curves 

There's a lot more we could say about B-Splines (what happens if we mess with the knots? How do you decide how high the degree should be? How do B-Splines work in 2 or more dimensions?), but that's beyond the scope of this article. For those interested, see: 
* [Shape Interrogation for Computer Aided Design and Manufacturing, Chapter 1](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node15.html), MIT Hyperbook
* [Definition of a B-Spline Curve](https://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/B-spline.pdf), UC Lecture notes
* [Desmos B-Spline Playground](https://www.desmos.com/calculator/ql6jqgdabs)
* and of course [B-Spline](https://en.wikipedia.org/wiki/B-spline), Wikipedia

**In conclusion: B-Splines are functions that let us trace arbitrary curves. To determine the value of the spline at some point `x`:**
1. **evaluate each basis function (which is a recursive function) at `x`**
2. **multiply by the basis function outputs by their corresponding conrtrol points**
3. **sum the results**

