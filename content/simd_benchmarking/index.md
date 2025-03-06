+++
date = '2025-03-04T19:00:59-05:00'
draft = false
title = 'Benchmarking Different Vectorization Strategies in Rust'
+++

{{< toc >}}


In this post, we'll go over what Single Instruction Multiple Data (SIMD) operations are and why you should care. We'll go over different ways to get SIMD operations included in your code, and we'll implement and benchmark those different methods to see how they work in practice. For our case study, we'll be calculating B-Splines. We'll briefly go over the math behind B-Splines, show a straightforward reference implementation, and then dive into how to optimize that implementation with SIMD operations. We'll measure how fast each method is, troubleshoot issues, and show how expected speed-ups can fail to materialize.

While this walkthrough uses Rust, the majority of what we discuss generalizes to other languages. If you’re familiar with writing SIMD Rust code and are just interested in seeing this particular example, skip to [Evaluating a B-Spline with Rust](#evaluating-a-b-spline-with-rust). 
 
This post is primarily a case-study. If you'd like to read about designing SIMD algorithms from scratch, I’ll direct you to McYoung’s delightful explainer at https://mcyoung.xyz/2023/11/27/simd-base64/

All work shown was conducted on an AWS [C7i](https://aws.amazon.com/ec2/instance-types/c7i/) EC2 instance running an [Intel 4th Gen Xeon processor](https://en.wikipedia.org/wiki/Sapphire_Rapids)

All graphics were generated using matplotlib in a Jupyter notebook, which can be downloaded <a href="/notebooks/simd_blogpost_graphics.ipynb" download="simd_benchmarking.ipynb"> here</a>

## What is Vectorization? An Intro to SIMD

Think of your CPU like a motorcycle. Fast, efficient, adaptable; good at weaving between alley-ways and through traffic. If you were going on a scavenger hunt all around a city, it’s exactly the tool you would want. But if you’re carrying packages from Houston to Austin, a motorcycle isn't the best tool. Sure, it’s fast, but even going felony-speeds that’s a 4 hour round-trip. Delivering half-dozen-packages would take a full 24 hours.

If you already know all the packages are coming from one place and going to another, you’re not going to use a motorcycle to deliver them, you're going to use a truck. Sure, it takes a little longer to make the trip, but when it can deliver hundreds of packages at once, as opposed to the single package at a time the motorcycle can manage, you're ultimately saving time. Of course if we’re running from point-to-point around town, the truck might not be the best choice.

Back to computers, the motorcycle is your CPU, and the truck is your GPU. Your CPU can perform operations very quickly, but there is some overhead. For every operation it’s got to pull in the instruction and the data, perform the operation, possibly store the results, and then go back for more. If the next operation depends on the results of the current one, then it’s the best tool you’ve got. But, if you know all your packages are going to the same place - that is to say, if you have a lot of data, and you want to do the same operation on all of it - then you’re better off running on a GPU, the 18-wheeler to your CPU’s motorcycle.

But sometimes you don’t want to run your code on a GPU. Maybe you don’t have one, maybe you don’t feel like learning how to writing GPU code, or maybe you just don’t have enough data to make it worth it. If your CPU is the motorcycle delivering one package at a time, and your GPU is an 18-wheeler delivering hundreds, what do you do when you have a handful of packages? Then you turn to… a motorcycle with a side car! It turns out your CPU actually has some parallel processing capabilities like your GPU does (assuming it's no more than a decade old or so). These operations, called Single Instruction Multiple Data (SIMD) or sometimes vector operations, take multiple arguments in parallel and perform the same operation to each one. For example, doing an element-wise add across two lists of numbers. 

Under normal conditions your processor would grab the first number from each list, add them together, write the result back to memory, then repeat on the next set of numbers. 

![A diagram demonstrating adding 4 pairs of numbers together, one after the other](generated_images/scalar_adding.png)

Using SIMD operations, the CPU can pull several numbers at a time from each list, add each chunk together, and write the entire chunk back at once 

![a diagram demonstrating adding 4 pairs of numbers together, all at once](generated_images/vector_adding.png)

In this article, we're going to show some straight-forward code to do some fancy mathematical calculations, and then we're going to "vectorize" it: turn it into SIMD code that will exercise this rarely-explicitly-used circuitry to run even faster. We'll explore a couple different options for doing so.

The next section talks about different ways to get your compiled Rust code to include SIMD operations, and after that we’ll focus on a particular algorithm which we'll optimize with SIMD instructions.

## Different Ways to Vectorize in Rust

For rest of this post, we’re going to be working in Rust. If you’re not a Rust developer, there will still be useful information here about how SIMD programming (and the pitfalls-thereof). We’ll be looking at three different ways to get our Rust code compiled into SIMD operations: 1) The compiler’s [auto-vectorization](https://llvm.org/docs/Vectorizers.html) 2) Rust’s [portable SIMD](https://doc.rust-lang.org/std/simd/index.html) module, and 3) CPU-specific intrinsics, focusing on a common 64-bit Intel x86 processor. Each method will have its own performance/portability/ease-of-use trade offs

### Auto-Vectorization: Let the Compiler Do the Work
The Rust compiler - alongside GCC, and any compiler based on LLVM like Clang - will do its best to turn your regular old code into fancy SIMD code for you. Whenever you compile with optimizations, the compiler does its best to identify portions that perform the same operation on sequential bits of data and speed things up with vector operations - for example, walking a pair of arrays and multiplying the elements together, or counting the number of times an element of the first array is larger than its companion in the second array. 

The compiler won't always recognize vectorizable code, though. The exact process by which a compiler determines what it can vectorize will vary between compilers and between versions. But, we can help the compiler help us by, in performance-sensitive parts of code we'd like the compiler to vectorize, minimizing conditionals and minimizing function calls. Conditionals and function calls make it harder for the compiler to "see" what our code is doing at the grand scale, which inhibits its ability to optimize it. That's not to say that a single If/Else kills auto-vectorization dead; but if you have a complicated loop you're trying to optimize, simplifying the branches may help more than you expect, thanks to the power of auto-vectorization!

### Rust's Portable SIMD module

Rust's handy-dandy [portable SIMD module](https://doc.rust-lang.org/std/simd/index.html) gives us the ability to write generic SIMD code, and then leave it to the compiler to figure out how to translate our SIMD operations into vector operations supported by the target CPU. If we're building for a CPU with no SIMD - fairly rare for most general-purpose computing applications, but common in embedded programming - the Rust compiler will even translate our code into plain-old scalar operations!

The only catch is the portable SIMD module is still considered an experimental API, and so it requires using [nightly Rust](https://doc.rust-lang.org/book/appendix-07-nightly-rust.html), as opposed to the default stable Rust. I've never had run into any stability issues with nightly Rust, but some organizations may be unwilling to accept the risk.

### CPU Intrinsics

The last method we'll explore to add SIMD operations to our code is with CPU intrinsics. In this method, we'll add special function calls to our code that the compiler will recognize as specific CPU operations we'd like to utilize. This method is similar to writing [inline assembly](https://doc.rust-lang.org/reference/inline-assembly.html), but with the added benefit that we can rely on the compiler completely to figure out how best to use available registers

## The Algorithm We'll Vectorize
Ok, let’s talk about the actual algorithm we want to speed up with SIMD operations: we’ll be calculating the value of [B-Splines](https://en.wikipedia.org/wiki/B-spline). If you’ve worked in CAD or digital graphics products, you may have heard the term, and know it as “that tool that lets me draw curvy lines by moving points around, even though the line doesn’t go exactly through the points”. In general, [Splines](https://en.wikipedia.org/wiki/Spline_(mathematics)) are piece-wise polynomial functions known for their ability to trace out arbitrary curves, and B-Splines are a specific way to define and build splines. If you’re unfamiliar with the math behind B-splines, here’s a brief primer, so you can understand the code later

### A Brief Primer on B-Splines
A B-spline is a recursive piece-wise function defined using three values
1. A list of numbers, called "knots", which define the intervals considered by the piece-wise function
2. A second list of numbers, called "control points" (or coefficients), which weight the different pieces of the function
3. A single number called the "degree" of the spline, which determines how many levels of recursion the function uses and, consequently, how smooth the resulting curve is

Let's look at an example:
![A degree-0 B-spline](generated_images/bspline_degree_0.png)

This is a dirt-simple degree-0 b-spline. The value of the spline at some value `x` is equal to the weighted sum of the constituent "basis functions" at `x`, so long as `x` is within the range defined by the knots; the function is zero everywhere else. In the above example, the weights (or “control points”) are all 1 for simplicity. Let’s take a look at some of the basis functions for this degree-0 spline

![the 1st basis function for a degree-0 B-spline](generated_images/degree_0/degree_0_basis_0.png)
![the 2nd basis function for a degree-0 B-spline](generated_images/degree_0/degree_0_basis_1.png)
![the 3rd basis function for a degree-0 B-spline](generated_images/degree_0/degree_0_basis_2.png)

Each degree 0 basis function is simply defined as `1` when `x` is between the `i'th` and `i+1'th` knot, and 0 everywhere else. Ok, so far, so boring. We’re just looking at some lines. Let’s start looking at higher degree B-splines to see how it comes together. Here’s the general form of the basis function for degree 1 and higher. It looks complicated, but we can break it down

![The basis function formula for B-splines degree 1 and higher](generated_images/basis_formula.png)

In english:
1. The `i'th` basis function of some degree `k` is equal to…
   1. The weighted combination of…
      1. The `i'th` basis function of degree `k-1`
      2. And…
      3. The `i+1'th` basis function of degree `k-1`
      * (remember that the 0th degree basis functions are just 1 or 0 as defined above, so `k=0` is the bottom layer)
   2. Where the weights are…
      1. Based on the “distance” between `x` and…
         1. The `i'th` knot, for the “left” lower-degree basis function
         2. The `i+k'th` knot, for the “right” lower-degree basis function
      2. Normalized by the length of the interval between the 
         1. `i+k'th` and `i'th` knot on the left
         2. `i+k+1'th` and `i+1'th` knot on the right 

That’s a lot of math. Let’s look at in action. Here are three degree-1 basis functions, along with the degree-0 basis functions on which they depend

![the 1st basis function for a degree-1 B-spline](generated_images/degree_1/degree_1_basis_0.png)
![the 2nd basis function for a degree-1 B-spline](generated_images/degree_1/degree_1_basis_1.png)
![the 3rd basis function for a degree-1 B-spline](generated_images/degree_1/degree_1_basis_2.png)

The 0th degree-1 basis function `B_0_1` "blends" the 0th and 1st degree-0 basis functions (`B_0_0` and `B_1_0` respectively) and so is non-zero where either `B_0_0` or `B_1_0` are non-zero, and zero everywhere else. Likewise `B_1_1` blends `B_1_0` and `B_2_0`, and so `B_1_1` is only non-zero over the range either of its dependencies are non-zero. 

Now, for degree-2 basis functions: 

![the 1st basis function for a degree-2 B-spline](generated_images/degree_2/degree_2_basis_0.png)
![the 2nd basis function for a degree-2 B-spline](generated_images/degree_2/degree_2_basis_1.png)
![the 3rd basis function for a degree-2 B-spline](generated_images/degree_2/degree_2_basis_2.png)

Again, each basis function is based on the ones below it. Each function `B_i_2` "blends" the basis functions `B_i_1` and `B_i+1_1`. All of these basis functions follow the formula defined above, where `k` equals the “degree” of the basis function, and `i` is given the value 1, 2, or 3 for the first, second, and third images in each set, respectively. The pattern would continue as we move to higher degrees - 3, 4, 5, and up. 

Let’s move up to a degree-3 b-spline and put all the basis functions together

![A full degree-3 B-spline with control points all set to 1](generated_images/bspline_degree_3_full.png)

The colored lines are each one of our basis functions, and the black line is the full B-spline. At any point `x`, the value of `spline(x)` is the sum of the values of each basis function `B_i` evaluated at that point `x`. In the above example the control points are all set to 1. Let's see another example with different control points to see how they affect things

![A full degree-3 B-spline with varying control points ](generated_images/bspline_degree_3_full_with_control_points.png)
<!-- {{< figure src="generated_images/bspline_degree_3_full_with_control_points.png" alt="A graph showing how adjusting the control points warps the curve traced by a B-Spline" caption="A degree-3 B-Spline with control points -0.5, 1, 0.5, and 1.2" >}} -->

Now we're cooking with gas! Here we see a B-spline in all it's glory. By manipulating the control points, we can "tug" portions of the spline curve in one direction or another. 

Through the proper choice of knots, control points, and degree, we can use B-Splines to construct arbitrary curves, and thus approximate any function we want

There's a lot more we could say about B-Splines (what happens if we mess with the knots? How what difference does increasing the degree make? How do B-Splines work in 2 or more dimensions?), but that's beyond the scope of this post. For those interested, see: 
* [Shape Interrogation for Computer Aided Design and Manufacturing, Chapter 1.4](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node15.html), MIT Hyperbook
* [Definition of a B-Spline Curve](https://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/B-spline.pdf), UC Lecture notes
* [Desmos B-Spline Playground](https://www.desmos.com/calculator/ql6jqgdabs)
* and of course [B-Spline](https://en.wikipedia.org/wiki/B-spline), Wikipedia

**In conclusion: B-Splines are functions that let us trace arbitrary curves. To determine the value of the spline at some point `x`:**
1. **evaluate each basis function (which is a recursive function) at `x`**
2. **multiply by the basis function outputs by their corresponding control points**
3. **sum the results**

Now that that we have some grasp of the math we'll be doing, let's implement it in code

### Evaluating a B-Spline with Rust

Let's begin with a straight forward implementation of our spline math. This code is intentionally sub-optimal, so we'll have a baseline for our benchmarking

```rust {linenos=inline}
/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `x`
fn basis_activation(i: usize, k: usize, x: f64, knots: &[f64]) -> f64 {
    // If degree is 0, the basis function is 1 if the parameter is within the range of the knot, and 0 otherwise
    if k == 0 {
        if knots[i] <= x && x < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    // Otherwise, we compute compute basis functions one degree lower, and use them to compute the current basis function
    let left_recursion = basis_activation(i, k - 1, x, knots);
    let right_recursion = basis_activation(i + 1, k - 1, x, knots);

    // Compute the weights for the left and right "child" basis functions
    let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
    let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);

    // Combine the left and right basis functions with the computed weights to produce the value of the current basis function
    let result = left_coefficient * left_recursion + right_coefficient * right_recursion;
    return result;
}

/// Calculate the value of the B-spline at the given parameter `x`
pub fn b_spline(x: f64, control_points: &[f64], knots: &[f64], degree: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..control_points.len() {
        // compute the value of each top-level basis function, multiplying it by the corresponding control point, and sum the results
        result += control_points[i] * basis_activation(i, degree, x, knots);
    }
    return result;
}
```

The function `b_spline()` calculates the value of a B-Spline function - defined by the provided knots and control points - at the provided point `x`. `b_spline()` calls `basis_activation()` to get the value of each top-level basis function at `x`, multiplies the value by the corresponding control point, and returns the final sum. In order to calculate the value of a given basis function, `basis_activation()` calls itself recursively to find the value of each "child" basis function on degree lower; once it gets down to `degree = 0`, there are no more "children" basis functions to consider, and the function returns `1` or `0` depending on whether or not `x` is between the specific knots considered by that basis function.

We'll be benchmarking this code with Rust's built-in benchmarking tool [`cargo bench`](https://doc.rust-lang.org/cargo/commands/cargo-bench.html). Here's a quick look at our benchmarking code

```rust {linenos=inline}
##![feature(test)]
extern crate test;
use test::Bencher;

use rust_simd_benchmarking::b_spline;

// define the parameters for the B-spline we'll use in each benchmark
fn get_test_parameters() -> (usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    let spline_size = 100;
    let input_size = 100;
    let degree = 4;
    let control_points = vec![1.0; spline_size];
    let knots = (0..spline_size + degree + 1)
        .map(|x| x as f64 / (spline_size + degree + 1) as f64)
        .collect::<Vec<_>>();
    let inputs = (0..input_size)
        .map(|x| x as f64 / input_size as f64)
        .collect::<Vec<_>>();
    (degree, control_points, knots, inputs)
}

##[bench]
// benchmark evaluating a degree-3 B-spline with 20 knots and 16 basis functions, over 100 different input values
fn bench_recursive_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        for x in inputs.iter() {
            let _ = b_spline(*x, &control_points, &knots, degree);
        }
    });
}
```

We're making the spline much larger than the examples we went over above - degree 4 with 16 basis functions and 100 different `x` values. We want the calculations to take long enough that the benchmarker can get an accurate read - even when we speed everything up later. We also need the number of knots, basis functions, and input values to be large enough that we have headroom to optimize - this will make more sense as we explore different vectorization strategies. For now, let's see how long this benchmark takes

```zsh
$> cargo bench -q

...

running 1 test
test bench_recursive_method ... bench:     706,262.10 ns/iter (+/- 81,587.11)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured; 0 filtered out; finished in 0.22s
```

Looks like evaluating all 100 inputs takes about 800k nanoseconds, or roughly 0.8 milliseconds. Cool, we have a baseline. Now we can start optimizing

From this point on, as we explore different vectorization strategies, there are lots of little details that a competent programmer might overlook at first, but which can have a large impact on performance - inefficient memory allocation, redundant looping, etc. I went through rigamarole of write-profile-optimize when I first wrote these operations as part of my [KAN Library](https://crates.io/crates/fekan). I will quietly include the results of all those lessons-learned in the code I show going forward, because I want each iteration of the algorithm to be the best version of itself it can be.

## Optimization #1: From Recursion to Looping

To understand our first optimization, let's take a step back and consider how the value of `B_i_k`, the value of the `i'th` basis function of degree `k`, depends on the values of the basis functions of degree `k-1`

![A pyramid showing the dependency chain for the 0th basis function at degree 3](generated_images/basis_pyramid/single_basis_pyramid.png)

One thing to note is that in each layer, each basis function is depended on by the one above it, and the one above-and-to-the-left of it. Our actual B-splines depend on more than one top-level basis function, however, so let's look at a version of this pyramid with multiple basis functions in the top layer

![A pyramid showing the dependency chain for the 0th through 3rd basis function at degree 3](generated_images/basis_pyramid/multiple_basis_pyramid.png)

A basis function is never depended **on** by *any* basis function to its right, and a basis function never **depends** on *any* basis function to its left. With that, we can rewrite our spline function in a loop that reuses previously calculated values instead of throwing them away.

```rust {linenos=inline}
/// Calculate the value of the B-spline at the given parameter `x` by looping over the basis functions
pub fn b_spline_loop_over_basis(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    // setup our data structures to hold the intermediate and final results
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];

    for x in inputs {
        let x = *x;

        // For the current value of `x`, fill the basis activations vec with the value of the degree-0 basis functions
        for i in 0..knots.len() - 1 {
            if knots[i] <= x && x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
        }
        /* For each degree k, compute the higher degree basis functions, 
         using the value of the "child" basis functions that were stored in the previous iteration,
         overwriting the child values once they're no longer needed */
        for k in 1..=degree {
            for i in 0..knots.len() - k - 1 {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
            }
        }

        // Finally, compute the ultimate value of this B-Spline at `x` by multiplying each basis function by the corresponding control point, and summing the results
        let mut result = 0.0;
        for i in 0..control_points.len() {
            result += control_points[i] * basis_activations[i];
        }
        outputs.push(result);
    }
    return outputs;
}
```

Here's our first optimized spline calculator. Now that we're not recursing, there's no need for a separate basis function - we do all our calculations in this one spline function. We're also taking in a whole batch of input values to be processed at once, instead of only taking one at a time, for efficiency reasons that will be explained in a moment.

In order to calculate the final value of the spline at point `x`, we need the value of each of our top level basis functions. To start, in that `0..knots.len() - 1` loop, we calculate the value of each degree-0 basis functions and store the results in a vector. 

![degree-3 pyramid of basis functions with all but the bottom layer greyed out](generated_images/basis_pyramid/multiple_basis_pyramid_bot_layer_filled.png)

Next, the `1..=degree` loop starting on line 26 is where the magic happens. At each layer `k`, starting at `1` and moving up to our full degree, we walk our vector of basis functions and calculate each in turn, overwriting the value of the lower-degree basis function that was in its spot. This works because of the direction of the arrows in the dependency pyramid. For example, when `k=1` and `i=0`, we're calculating basis function `B_0_1`, which depends on `B_0_0` and `B_1_0`, which at that point live in our vector at the `0th` and `1st` position, respectively. We read those values from the vector, and use them to calculate `B_0_1`

![degree-3 pyramid of basis functions with all but the bottom layer greyed out. There's a red box around the first basis function in the second layer](generated_images/basis_pyramid/calculating_B_0_3.png)

Then we write `B_0_1` to the `0th` position in our vector, overwriting `B_0_0`, which is no longer needed. After that we move on to calculating `B_1_1`

![degree-3 pyramid of basis functions. The first basis function in the second layer is filled in, as are the second-through-last basis functions in the bottom layer. The rest are greyed out. There's a red box around the second basis function in the second layer](generated_images/basis_pyramid/calculating_B_1_3.png)

Which overwrites `B_1_0`, and so on

![degree-3 pyramid of basis functions. The first and second basis function in the second layer are filled in, as are the third-through-last basis functions in the bottom layer. The rest are greyed out. There's a red box around the third basis function in the second layer](generated_images/basis_pyramid/calculating_B_2_3.png)

![degree-3 pyramid of basis functions. The first-through-third basis function in the second layer are filled in, as are the fourth-through-last basis functions in the bottom layer. The rest are greyed out. There's a red box around the fourth basis function in the second layer](generated_images/basis_pyramid/calculating_B_3_3.png)

Each iteration of that second loop fills in one layer of our pyramid. Once we finish, the first several elements of our `basis_activations` vector are the outputs of our top-level basis functions; the remaining values are leftover basis outputs from lower levels that were never overwritten, and can be safely discarded

The `0..control_points.len()` loop at the end should look familiar - we're just summing each basis function multiplied by its control point, as before.

Now, let's see how fast this method is

```zsh
>$ cargo bench -q

running 4 tests
iiii
test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 2 tests
test bench_recursive_method   ... bench:     807,886.80 ns/iter (+/- 37,114.71)
test bench_simple_loop_method ... bench:      64,020.88 ns/iter (+/- 1,374.68)

test result: ok. 0 passed; 0 failed; 0 ignored; 2 measured; 0 filtered out; finished in 2.50s
```
(Ignore the changes in the time for the recursive method benchmark - the benchmarker is imperfect, and there will always be some minor variation between runs)

This method takes about 64k nanoseconds. Looks like we got over a 10x speedup just by moving from recursion to looping! That speedup is coming, in varying degrees from three places:
1. Reduced overhead. Besides the work done within a function, there's a certain amount of work required simply to call a function and return from it. Our recursive method had a lot of function calls - now we have only one
2. Reusing calculated basis values. Go back and look at our pyramid of basis functions; each basis function `B_i_k` is depended on by two other basis functions - `B_i-1_k+1` and `B_i_k+1`. In the recursive method, we'd calculate `B_i_k` once while calculating its first dependent, and again when calculating its second dependent. Now that we're storing the results of each basis function calculation in our vector, we only need to calculate each one once
3. **Auto-vectorization**. In recursive mode, the compiler was limited in what it could assume about our code, so it was forced to be conservative in how it optimized and wrote assembly to do exactly what we described and nothing more - read a few values, multiply and add them together, and give a single value back. Now that we're working a loop, the compiler is able to recognize that we're walking a vector and performing the same operation at each step, and do things smarter: the compiler is generating assembly with SIMD operations. While our Rust code says "for each index 0..n, read a few values, multiply and add them together, and store the single result", the assembly generated by the compiler now says "for every chunk of indexes [0..i]...[n-i..n], read several chunks of values, multiply and add the chunks together, and store the several results all at once". We're getting vectorization for free, just by writing code that's easier for the compiler to understand!


## Optimization #2: Rust's Portable SIMD Crate

Now we'll start introducing SIMD operations using Rust's [portable SIMD module](https://doc.rust-lang.org/std/simd/index.html). 

Note for those following along with their own code at home: using `std::simd` requires adding the `#![feature(portable_simd)]` flag at the top of our library and compiling with the nightly toolchain, instead of the default stable release. You can install the `nightly` toolchain using [rustup] (https://www.rust-lang.org/tools/install) with `rustup toolchain install nightly`, and set it as the default toolchain for your project by calling `rustup override set nightly` from within your project directory

Below is our B-spline calculation function using SIMD operations. It calculates everything the same way as our looping method, but uses explicit SIMD calls to operate on multiple elements at the same time

```rust {linenos=inline}
const SIMD_WIDTH: usize = 8;

pub fn b_spline_portable_simd(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::simd::prelude::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];

    for x in inputs {
        let x_splat: Simd<f64, SIMD_WIDTH> = Simd::splat(*x);
        // fill the basis activations vec with the value of the degree-0 basis functions
        let mut i = 0;
        while i + SIMD_WIDTH < knots.len() - 1 {
            let knots_i_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i..]);
            let knots_i_plus_1_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + 1..]);

            let left_mask: Mask<i64, SIMD_WIDTH> = knots_i_vec.simd_le(x_splat); // create a bitvector representing whether knots[i] <= x
            let right_mask: Mask<i64, SIMD_WIDTH> = x_splat.simd_lt(knots_i_plus_1_vec); // create a bitvector representing whether x < knots[i + 1]
            let full_mask: Mask<i64, SIMD_WIDTH> = left_mask & right_mask; // combine the two masks
            let activation_vec: Simd<f64, SIMD_WIDTH> =
                full_mask.select(Simd::splat(1.0), Simd::splat(0.0)); // create a vector with 1 in each position j where knots[i + j] <= x < knots[i + j + 1] and zeros elsewhere
            activation_vec.copy_to_slice(&mut basis_activations[i..]); // write the activations back to our basis_activations vector

            i += SIMD_WIDTH; // increment i by SIMD_WIDTH, to advance to the next chunk
        }
        // since knots.len() - 1 is not guaranteed to be a multiple of SIMD_WIDTH, we need to handle the remaining elements one by one
        while i < knots.len() - 1 {
            if knots[i] <= *x && *x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
            i += 1;
        }

        // now to compute the higher degree basis functions
        for k in 1..=degree {
            let mut i = 0;
            while i + SIMD_WIDTH < knots.len() - k - 1 {
                let knots_i_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i..]);
                let knots_i_plus_k_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + k..]);
                let knots_i_plus_1_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&knots[i + 1..]);
                let knots_i_plus_k_plus_1_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&knots[i + k + 1..]);

                // grab the value for and calculate the coefficient for the left term of the recursion, doing a SIMD_WIDTH chunk at a time
                let left_coefficient_vec =
                    (x_splat - knots_i_vec) / (knots_i_plus_k_vec - knots_i_vec);
                let left_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[i..]);

                // grab the value for and calculate the coefficient for the right term of the recursion, doing a SIMD_WIDTH chunk at a time
                let right_coefficient = (knots_i_plus_k_plus_1_vec - x_splat)
                    / (knots_i_plus_k_plus_1_vec - knots_i_plus_1_vec);
                let right_recursion_vec: Simd<f64, SIMD_WIDTH> =
                    Simd::from_slice(&basis_activations[i + 1..]);

                let new_basis_activations_vec = left_coefficient_vec * left_recursion_vec
                    + right_coefficient * right_recursion_vec;
                new_basis_activations_vec.copy_to_slice(&mut basis_activations[i..]);

                i += SIMD_WIDTH;
            }
            // again, since knots.len() - k - 1 is not guaranteed to be a multiple of SIMD_WIDTH, we need to handle the remaining elements one by one
            while i < knots.len() - k - 1 {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
                i += 1;
            }
        }

        // now to compute the final result, in chunks of SIMD_WIDTH
        let mut i = 0;
        let mut result = 0.0;
        while i + SIMD_WIDTH < control_points.len() {
            let control_points_vec: Simd<f64, SIMD_WIDTH> = Simd::from_slice(&control_points[i..]);
            let basis_activations_vec: Simd<f64, SIMD_WIDTH> =
                Simd::from_slice(&basis_activations[i..]);
            result += (control_points_vec * basis_activations_vec).reduce_sum();
            i += SIMD_WIDTH;
        }
        // handle the remaining elements one by one
        while i < control_points.len() {
            result += control_points[i] * basis_activations[i];
            i += 1;
        }
        outputs.push(result);
    }

    return outputs;
}
```

We've gone from a little over 30 lines, to a solid 100 lines of code! Since we can't guarantee ever that the number of basis functions we're updating is an exact multiple of our SIMD window, after every SIMD-loop we need a regular scalar loop to handle the remainder. 

Let's see how much explicitly using SIMD operations has sped up our benchmarks

```zsh
$> cargo bench -q

running 4 tests
iiii
test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 3 tests
test bench_portable_simd_method ... bench:      55,866.81 ns/iter (+/- 984.44)
test bench_recursive_method     ... bench:     800,009.20 ns/iter (+/- 20,489.24)
test bench_simple_loop_method   ... bench:      63,576.17 ns/iter (+/- 1,390.29)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out; finished in 5.74s
```
we went from 64k nanoseconds to 56k nanoseconds, or about a 1.15x speedup. That's not nothing, but it's not great either. It's a little faster, suggesting that writing out the SIMD operations explicitly did in fact help, but we were definitely hoping for more... Let's take a look under the hood at the assembly, and make sure we're getting the SIMD operations we expect. 

There are a lot of tools available to inspect assembly - for this investigation I used [ghidra](https://ghidra-sre.org). Let's see what our `k in 1..=degree` loop looks like once it's compiled.

![The k>=1 basis calculation loop, compiled with `cargo build -r`](images/portable_first_look.png)

On the left we have the main workhorse loop of our spline calculations, and on the right is a portion of the assembly code for that loop. We don't need to go through the assembly in detail, but one thing jumps out immediately: **we're only using 128-bit SIMD operations, instead of the expected 512-bit**. In [x86 assembly SIMD operations](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), the `XMM` mnemonic is used to refer to 128-bit registers; `YMM` refers to 256-bit registers, and `ZMM` refers to 512-bit registers. From this we can conclude that the compiler only generated assembly code to utilize the processor's 128-bit SIMD capabilities, and did not generate assembly code to use the processor's full 512-bit SIMD capabilities.

In Rust's portable SIMD module, it's incumbent on the programmer to denote explicitly how "wide" of SIMD operations they want to use - do you want to operate on 2 values simultaneously, or 4, or 8? In our code we define the constant `SIMD_WIDTH`, which we set equal to `8` and use to tell the Rust SIMD code how many values we want to pack together. Since our code says to pack together 8 values, and we're working with 64-bit floats, and 8 x 64 = 512, we'd expect to see `ZMM` littered throughout our assembly. But it's missing.

It turns out, this is a feature, not a bug. Recall that we're using the Rust's **portable** SIMD module - by design, if the CPU for which we're compiling can't handle any of the SIMD operations we've requested, the compiler will replace them with operations the CPU *can* handle. Even though [most modern x86 CPUs](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) have 512-bit registers, not every x86 CPU in existence has the circuitry to perform 512-bit operations, and so by default the rust compiler will assume 512-bit operations aren't available. To prove to ourselves this is true, we can get `rustc` (the rust compiler) to tell us what sort of machine it's compiling for, and what CPU features it believes are available

```zsh
>$ rustc -vV | grep host
host: x86_64-unknown-linux-gnu
>$ rustc --print cfg | grep feature
target_feature="fxsr"
target_feature="sse"
target_feature="sse2"
target_feature="x87"
```

Despite the fact that I'm running on an [Intel 4th gen Xeon](https://aws.amazon.com/ec2/instance-types/c7i/) processor, which [absolutely has](https://en.wikipedia.org/wiki/Sapphire_Rapids) the AVX-512 feature (and thus 512-bit capabilities), the compiler is targeting a generic x86 CPU, and believes it can only use up to SSE and SSE2 feature sets (which explains the `XMM` registers we saw in the assembly code). In order to use the full feature set of our processor, we need to tell the compiler specifically what sort of processor it ought to compile for. We do this with the [`target-cpu` flag](https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu). 

The CPU architecture for our 4th gen Xeon is called Sapphire Rapids; let's ask the compiler what features it thinks a Sapphire Rapids CPU has.

```zsh
$> rustc --print cfg -Ctarget-cpu=sapphirerapids | grep feature
...
target_feature="avx"
target_feature="avx2"
target_feature="avx512bf16"
target_feature="avx512bitalg"
target_feature="avx512bw"
target_feature="avx512cd"
target_feature="avx512dq"
target_feature="avx512f"
target_feature="avx512fp16"
target_feature="avx512ifma"
target_feature="avx512vbmi"
target_feature="avx512vbmi2"
target_feature="avx512vl"
target_feature="avx512vnni"
target_feature="avx512vpopcntdq"
target_feature="avxvnni"
...
target_feature="sse"
target_feature="sse2"
target_feature="sse3"
target_feature="sse4.1"
target_feature="sse4.2"
target_feature="ssse3"
...
```

The compiler knows that a Sapphire Rapids CPU can handle the full range of AVX-512 operations, so we just need to tell the compiler that it should in fact compile for Sapphire Rapids, by passing `-Ctarget-cpu=sapphirerapids` in when we compiled (you can also use `-Ctarget-cpu=native` to tell the compiler "target whatever CPU you're currently on". I'll stick with the former throughout this text for clarity). We need pass the flag to the compiler through the `RUSTFLAGS` environment variable since we're calling `cargo` instead of calling `rustc` directly. 

Let's recompile our code and take another look in Ghidra
![the k>=1 basis calculation loop, compiled with `RUSTFLAGS="-C target-cpu=sapphirerapids" cargo build -r`](images/portable_target_cpu.png)

*Now* we see the `ZMM` register usage we expect! We've successfully convinced the compiler to take full advantage of the 512-bit circuitry present in our CPU. Since we've 4x'd the size of the SIMD operations used by our program (moving from 128-bit `XMM` registers to 512-bit `ZMM` registers), we should expect close to a 4x speedup!

```zsh
$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo bench -q

running 4 tests
iiii
test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 3 tests
test bench_portable_simd_method ... bench:      56,091.13 ns/iter (+/- 2,762.90)
test bench_recursive_method     ... bench:     802,534.40 ns/iter (+/- 40,154.44)
test bench_simple_loop_method   ... bench:      66,307.14 ns/iter (+/- 2,133.62)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out; finished in 2.79s
```

We got... no speedup? Not only that, it's actually a little bit **slower**. This is quite counter-intuitive, and deserves additional investigation. Let's go through some reasons we might see this performance non-change, and try and rule out as many as we can.

Broadly speaking, there are three things that could be going wrong:
1. Cold SIMD code. A portion of code is said to be "cold" if it's used infrequently during the course of normal operations. It could be that we improved part of the program that only accounts for a small portion of the total runtime
2. CPU downclocking. The CPU could be slowing itself down to help deal with the extra heat generated by 512-bit operations, which it turns out [is a thing that happens](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Downclocking)
3. CPU Bottleneck-ing. We were able to get our code do more calculations with fewer instructions, but we hit a bottleneck somewhere limiting how fast we can get through instructions

The next few sections investigate our lackluster SIMD performance and get pretty in the weeds (and spoiler alert: we only ever get to a partial explanation). Those uninterested in walking through CPU internals and how they affect performance, and who are satisfied with the answer "Moving from 128-bit operations to 512-bit operations doesn't always produce the speedup you're expecting" can safely skip over the next few sections and go to [Optimization #3](#optimization-3-x86-intrinsics)

### SIMD Slowdown Hypothesis 1: Cold Code

First things first - are we actually *using* the SIMD loops enough for the faster calculations to matter? We expect the SIMD-using `1..=degree` loop to be "hot", meaning we spend a significant chunk of runtime there, and thus speeding up the loop should speed up the overall program. But, maybe those loops *are* running faster, but they're actually "cold" - they represent a small fraction of our overall runtime, so we don't notice any speed ups. Let's really crank the size of the calculations we're benchmarking and see if that reveals a greater difference between the version compiled for a generic CPU, and the version compiled for a Sapphire Rapids CPU with AVX-512 operations

```rust {linenos=inline}
// define the parameters for the B-spline we'll use in each benchmark
fn get_test_parameters() -> (usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    let spline_size = 1000; // increased from 100 to 1000
    let input_size = 100;
    let degree = 4;
    let control_points = vec![1.0; spline_size];
    let knots = (0..spline_size + degree + 1)
        .map(|x| x as f64 / (spline_size + degree + 1) as f64)
        .collect::<Vec<_>>();
    let inputs = (0..input_size)
        .map(|x| x as f64 / input_size as f64)
        .collect::<Vec<_>>();
    (degree, control_points, knots, inputs)
}
```

We increased the `spline_size` value - which controls the number of basis functions - from 100 to 1000. Let's see if 10x-ing the time we spend calculating basis functions reveals a performance difference. First, the generic version, as a baseline:
```zsh
$> cargo bench -q

...

running 3 tests
test bench_portable_simd_method ... bench:     476,557.90 ns/iter (+/- 17,461.56)
test bench_recursive_method     ... bench:   7,652,735.90 ns/iter (+/- 691,527.68)
test bench_simple_loop_method   ... bench:     549,506.40 ns/iter (+/- 9,020.09)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out; finished in 2.64s
```

And now, including AVX-512 operations:

```zsh
$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo bench -q

...

running 3 tests
test bench_portable_simd_method ... bench:     470,703.35 ns/iter (+/- 8,116.68)
test bench_recursive_method     ... bench:   7,819,541.70 ns/iter (+/- 497,504.42)
test bench_simple_loop_method   ... bench:     551,491.70 ns/iter (+/- 10,054.52)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out; finished in 2.83s
```

From **477k +/- 17k** to **471k +/- 8k**. There's a *bit* of a difference, but within the margin of error. Certainly not the 4x speed increase we were expecting. 

**Conclusion: The disappointing performance is not caused by cold SIMD loops.** One hypothesis down, two to go!

### SIMD Slowdown Hypothesis 2: Dastardly Downclocking

One interesting fact I was surprised to learn when I started SIMD programming: CPUs will reduce their clock speed for [programs](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/) with AVX instructions. Since these instructions operate on more data at once, they consume more power, and thus generate more heat-per-unit-time than the processor usually has to handle; it makes sense the processor might need to slow down in response.

Now, this probably isn't the problem: downclocking will most hurt programs that spend only a little bit of time doing AVX-512 operations and a lot of time doing regular scalar operations, which would take the hit from the slower clock speed. Since we just showed that our problem isn't too little time spent in the SIMD loop, we're probably not hurting from reduced clock speed. Nonetheless, let's rule try and rule it out.

Turns out this hypothesis is really easy to test. We'll use a linux built-in tool [perf](https://man7.org/linux/man-pages/man1/perf.1.html) - specifically [perf-stat](https://man7.org/linux/man-pages/man1/perf-stat.1.html) - to check the clock speed for our two different versions. 

We're going to add a simple `main` function to our code so we can run it on its own, instead of running it as part of a benchmark. Since we're measuring with `perf` instead of `cargo bench`, this will help us isolate and measure *only* the code we're testing, and keep any code added by the benchmarking infrastructure from muddying our measurements.

Here's the `main` function, for reference. It does exactly what the benchmark was doing: calculating different basis values for each of the different inputs. We're increasing the number of calculations done to ensure the code runs long enough for perf to get a good measurement
```rust {linenos=inline}
pub fn main() {
    let spline_size = 2000;
    let input_size = 2000;
    let degree = 4;
    let control_points = vec![1.0; spline_size];
    let knots = (0..spline_size + degree + 1)
        .map(|x| x as f64 / (spline_size + degree + 1) as f64)
        .collect::<Vec<_>>();
    let inputs = (0..input_size)
        .map(|x| x as f64 / input_size as f64)
        .collect::<Vec<_>>();
    let _ =
        rust_simd_benchmarking::b_spline_portable_simd(&inputs, &control_points, &knots, degree);
}
```

Ok, let's see if the clock speed changes when we move from using the default 128-bit operations to the advanced 512-bit operations. We'll use perf to count the number of clock-cycles and the amount of time our program took to complete, and perf will helpfully give us the clock speed in GHz. If downclocking is our culprit, we expect to see a reduction in clock speed from our default version to our AVX-512 version. Let's look - default version is first:

```zsh
$> cargo build -r
   Compiling rust_simd_benchmarking v0.1.0 (/home/ec2-user/spline_simd_benchmarking)
    Finished `release` profile [optimized + debuginfo] target(s) in 0.14s
$> perf stat -e cycles,cpu-clock target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72700092      cycles:u                         #    3.329 GHz                    
             21.84 msec cpu-clock:u                      #    0.157 CPUs utilized          

       0.138916195 seconds time elapsed

       0.092568000 seconds user
       0.046332000 seconds sys


$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo build -r
   Compiling rust_simd_benchmarking v0.1.0 (/home/ec2-user/spline_simd_benchmarking)
    Finished `release` profile [optimized + debuginfo] target(s) in 0.13s
$> perf stat -e cycles,cpu-clock target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72179853      cycles:u                         #    3.239 GHz                    
             22.29 msec cpu-clock:u                      #    0.160 CPUs utilized          

       0.139505178 seconds time elapsed

       0.092915000 seconds user
       0.046547000 seconds sys
```
The average clock speed for the default version is 3.329 GHz, and for the AVX-512 version it's 3.239 GHX. That's 3% slower, which is certainly not enough to counteract the 4x gain we expected to get from AVX-512.

**Conclusion: the disappointing performance is not due to CPU downclocking**

### SIMD Slowdown Hypothesis 3: Bad Bottleneck

That just leaves bottlenecks as our remaining hypothesis. First, a brief sanity check. The whole point of adding SIMD operations is to complete the same amount of work in fewer steps. Let's make sure we're actually doing fewer *operations*, even if the total *time* hasn't changed

```zsh
$> cargo build -r
    Finished `release` profile [optimized + debuginfo] target(s) in 0.00s
$> perf stat -e cycles,instructions target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          73051145      cycles:u                                                           
         250509449      instructions:u                   #    3.43  insn per cycle         

       0.137525791 seconds time elapsed

       0.091608000 seconds user
       0.045851000 seconds sys


$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo build -r
    Finished `release` profile [optimized + debuginfo] target(s) in 0.00s
$> perf stat -e cycles,instructions target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72016963      cycles:u                                                           
         109372993      instructions:u                   #    1.52  insn per cycle         

       0.139154079 seconds time elapsed

       0.092731000 seconds user
       0.046416000 seconds sys
```

In our default mode using 128-bit operations, our program takes 250M instructions. When we upgrade to 512-bit operations, it takes 109M instructions. So, we **ARE**, in fact, completing our work in fewer instructions.  It's more than the 1/4-as-many as one might expect moving from 128-bit to 512-bit operations (128 * 4 = 512), but since plenty of the operations aren't vector operations and don't actually change between our default and advanced versions, it makes sense that we wouldn't see a full 75% reduction. We revise our "expected" speedup from 4x down to ~2.3x. That's still a significant speedup that we're not seeing, so let's find out where it went.

We can also see above that the average Instructions Per Cycle (IPC) drops from 3.43 to 1.52. So the default version requires 2.3x as many operations, but it also is able to process (oh would you look at that) 2.3x times as many instructions per cycle! It makes sense now that our default and advanced versions would take the same amount of time in our benchmarks - but it still doesn't explain *why*.

(Off the top of my head, I would say this points to a CPU parallelism issue not a memory bottleneck; but, as I mentioned above, I'm writing this from the future where I already know the answer, so that may not be a fair 'prediction'.)

Somewhere, deep in the bowels of our processor, moving from 128-bit to 512-bit operations is causing us to get "stuck" waiting on something. Now for exactly what that *something* is, we have [plenty of options](images/Golden_Cove.png). Broadly speaking, we can divide the possible bottlenecks into two categories, which will help us search for evidence: memory bottlenecks and [instruction-level parallelism](https://en.wikipedia.org/wiki/Instruction-level_parallelism) (ILP) bottlenecks.

Memory bottlenecks: At the end of the day, how fast our processor can crunch numbers is limited by how fast it can pull those numbers from memory. CPU's have [multiple levels of caches](https://www.geeksforgeeks.org/multilevel-cache-organisation/) in order to keep needed data close by and reduce those bottlenecks, but even those caches only go so fast (and can only hold so much data). We may have improved the speed at which our program crunches numbers so much we've run up against the speed at which it can get new numbers to crunch. We'll use perf to measure the number of cycles our program spends waiting on data from different levels of the cache to see if that's what's holding us back 

Instruction-level Parallelism: CPU's already do a lot of work under the hood to maximize the number of completed instructions per cycle.
   1. [Instruction Pipelineing](https://en.wikipedia.org/wiki/Instruction_pipelining) allows parts of the CPU that handle the beginning, middle, and end of processing an instruction are all in use at once by starting work on the current instruction before work the previous instruction has totally finished. Imagine a conveyor belt in a car factory, with different stations for welding the frame, attaching the body, and painting; You complete more cars faster if you keep each station busy, rather than waiting for one car to be totally finished before starting the next.
   2. [CPU Superscaling](https://www.geeksforgeeks.org/superscalar-architecture/) let's the CPU work on multiple instructions at the same time, even if they're all in the same stage of the instruction pipeline. Imagine a choosy tollbooth: multiple lanes let multiple cars pass at the same time instead of waiting on each other; The catch is that each tollbooth can only take certain kinds of vehicles - that's the "choosy" part. CPUs distribute the circuitry that lets execute instructions - add numbers together, compare numbers, request data from memory etc. - across multiple independent [execution units](https://en.wikipedia.org/wiki/Execution_unit) (EU), but each EU can only do a few different tasks. It may be that by moving to AVX-512 operations, our CPU instructions are now being routed to fewer EUs and backing up behind each other. Sort of like if we replaced all the different cars going through different lanes of our choosy tollbooth with a few busses that can all only go through one lane
   3. [Out of Order Execution](https://en.wikipedia.org/wiki/Out-of-order_execution#Basic_concept) enables later instructions that don't depend on the outcome of earlier instructions to go ahead and skip the line, instead of waiting. Imagine a waiting room full of patients filling out paperwork to see a doctor: you'll be able to help patients faster if you let them see a doctor as soon as they've completed their paperwork; if you insist the patients be seen in the order they came in, you may have several patients sitting and waiting on one particular patient who's slow with their paperwork
So, it may be that moving to AVX-512 operations for better data-level parallelism - processing more data per CPU instruction - is interfering with the CPU's ability to provide instruction-level parallelism - processing more instructions per cycle. Maybe our instructions tend to be more depended on the ones that came before, interfering with pipelining or out-of-order execution, or maybe the AVX-512 instructions are all being routed the same few EUs, preventing us from taking advantage of superscaling.

While we're generally capable of determining *if* we have an instruction-level parallelism problem, finding the exact cause would be difficult: the systems that provide ILP are so complex, a full discussion of each of them could literally [fill multiple college courses](https://student.mit.edu/catalog/m6a.html#6.1920). To make life even more difficult, my ability to check for ILP problems with `perf` is... currently inhibited. 

In a perfect world, I would use stats from `perf`'s Topdown Microarchitecture Analysis (TMA) family events; The TMA events are "smarter" counts that are synthesized from a number of different statistics and cut right to the heart of the matter. I would *like* to use measure `tma_core_bound`, `tma_memory_bound`, and `tma_frontend_bound` events to count the number of times the CPU wasted an opportunity to do useful work computation, memory access, and instruction fetch and decoding, respectively. With those measurements, we could immediately see if our bottleneck was happening reading and writing from memory (`tma_memory_bound`); or an ILP problem, either pulling new instructions into the CPU (`tma_frontend_bound`) or processing instructions within the CPU (`tma_core_bound`). *However*, because the universe is cruel and unjust, `perf stat` refuses to recognize those events, despite the fact that `perf lists` claims they are valid.

So! We'll have to pull back the covers and look at some more granular data to figure out where the bottleneck is! We'll be measuring the number of "stalls" that occur for one reason or another. A "stall" is counted every time a CPU instruction was prevented from moving forward in the processing pipeline because a resource it needed (data, available circuitry, etc.) was unavailable.

First, let's check for memory bottlenecks. We'll check the number of stalls due to cache misses and see how long our program has to sit and wait for data from a lower tier of memory. Every L1d stall represents an instruction that couldn't proceed that CPU cycle because the data it needed wasn't in the L1 cache and had to be retrieved from the L2 cache; L2 stalls represent having to wait for data from the L3 cache; and L3 stalls represent having to wait for data from memory

```zsh
$> perf stat -e cycles,instructions,cycle_activity.stalls_total,memory_activity.stalls_l1d_miss,memory_activity.stalls_l2_miss,memory_activity.stalls_l3_miss target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72678529      cycles:u                                                           
         250508875      instructions:u                   #    3.45  insn per cycle         
           3027025      cycle_activity.stalls_total:u                                      
            134077      memory_activity.stalls_l1d_miss:u                                   
            118043      memory_activity.stalls_l2_miss:u                                   
                 0      memory_activity.stalls_l3_miss:u                                   

       0.025591287 seconds time elapsed

       0.020625000 seconds user
       0.000000000 seconds sys


$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo build -r
    Finished `release` profile [optimized + debuginfo] target(s) in 0.02s
$> perf stat -e cycles,instructions,cycle_activity.stalls_total,memory_activity.stalls_l1d_miss,memory_activity.stalls_l2_miss,memory_activity.stalls_l3_miss target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72073554      cycles:u                                                           
         109373036      instructions:u                   #    1.52  insn per cycle         
          26101579      cycle_activity.stalls_total:u                                      
            167832      memory_activity.stalls_l1d_miss:u                                   
            134337      memory_activity.stalls_l2_miss:u                                   
                 0      memory_activity.stalls_l3_miss:u                                   

       0.023991590 seconds time elapsed

       0.020727000 seconds user
       0.000000000 seconds sys
```

Alright, we have zero L3 stalls. That makes some sense since we're only working with an amount of data significantly smaller than the [L3 cache can hold](https://www.tomshardware.com/news/intel-alder-lake-specifications-price-benchmarks-release-date). 

Looks like memory stalls grow about 20% when we introduce AVX-512 operations, which might be cause for alarm, but we can see both that memory stalls are a small portion of the overall stalls and that total stalls grew *significantly* faster than memory stalls - 800% from the default to AVX-512 versions. Memory stalls actually decrease as a percentage of total stalls from 8% in the default version to 1% in the AVX-512 version. It's fair to say, then, that **memory bottlenecks are not the reason for the disappointing AVX-512 performance**

Let's take a look at how well each version of our program utilizes CPU superscaling by measuring how many EUs each version of the program has in use, on average. The specific `perf` stats we'll use `exe_activity.x_ports_util`, which count how many cycles `x` execution ports were in use. For our purposes, you can consider "execution unit" and "execution port" synonymous. Our Golden Cove architecture actually has [12 execution ports](https://download.intel.com/newsroom/2021/client-computing/intel-architecture-day-2021-presentation.pdf#page=41) that could be used each cycle, but `perf` only has stats up to 4 in use at once. As best I can tell, `perf` is limited by the counters actually present in hardware, which are limited by cost/benefit trade-offs. We'll hope that counts of cycles with 5 or more execution ports in use are either included in `exe_activity.4_ports_util`, or are few enough to be insignificant

```zsh
$> cargo build -r
    Finished `release` profile [optimized + debuginfo] target(s) in 0.00s
$> perf stat -e cycles,instructions,exe_activity.exe_bound_0_ports,exe_activity.1_ports_util,exe_activity.2_ports_util,exe_activity.3_ports_util,exe_activity.4_ports_util target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72831716      cycles:u                                                           
         250508690      instructions:u                   #    3.44  insn per cycle         
            276409      exe_activity.exe_bound_0_ports:u                                   
          12387051      exe_activity.1_ports_util:u                                        
          18029924      exe_activity.2_ports_util:u                                        
          14596390      exe_activity.3_ports_util:u                                        
          10552752      exe_activity.4_ports_util:u                                        

       0.139553509 seconds time elapsed

       0.092941000 seconds user
       0.046571000 seconds sys


$> RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo build -r
    Finished `release` profile [optimized + debuginfo] target(s) in 0.00s
$> perf stat -e cycles,instructions,exe_activity.exe_bound_0_ports,exe_activity.1_ports_util,exe_activity.2_ports_util,exe_activity.3_ports_util,exe_activity.4_ports_util target/release/rust_simd_benchmarking

 Performance counter stats for 'target/release/rust_simd_benchmarking':

          72110720      cycles:u                                                           
         109373138      instructions:u                   #    1.52  insn per cycle         
           4623206      exe_activity.exe_bound_0_ports:u                                   
          23005987      exe_activity.1_ports_util:u                                        
          14070220      exe_activity.2_ports_util:u                                        
           7806487      exe_activity.3_ports_util:u                                        
           3115760      exe_activity.4_ports_util:u                                        

       0.145321678 seconds time elapsed

       0.096560000 seconds user
       0.048280000 seconds sys
```

![a graph showing the count of cycles that used a given number of execution ports](generated_images/port_utilization.png)

When graphed, it's obvious the AVX-512 version has reduced EU usage. We see that the default version uses 2.41 execution units on average, while the AVX-512 version only uses 1.65. Another way to read that is "during the course of the program, the default version tended to be working on 2.41 operations at once, while the AVX-512 version only tended to be working on 1.65 operations at once".

Now, the astute reader will notice that we were trying to account for a missing 2.3x speedup, but the default version only averages 1.46x as many execution ports as the AVX-512 version. Clearly, we've only partially explained why we're not seeing a speedup but we're going to leave things here because: 
* Finding the exact cause of bottlenecks would take a great deal more work, since there are [a number of places](https://commons.wikimedia.org/wiki/File:Golden_Cove.png) the CPU could get backed up, and probably require more specialized tools like [Intel's VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html#gs.k48rbh)
* Even if we knew the exact source of the bottleneck, it may not be something we can practically address in our code
* We'd be getting well beyond the scope of this article, which is supposed to be comparing different methods of adding SIMD operations to code, not an extensive exploration of Intel microarchitecture


So, we're going to conclude our investigation of Rust's portable SIMD module, and leave this section with the following conclusion: **Our attempts to speed up our program by adding data-level parallelism with Rust's portable SIMD module were stymied by reduced instruction-level parallelism caused by the nuances of our CPU's microarchitecture.**

Now, we can't say it's *never* useful to add data-parallelism with the Rust portable SIMD module, we can be certain it didn't help for our use case. Even so, you've now seen an example of how to add SIMD operations to code, benchmark, and troubleshoot issues. Next, we'll explore another way to add SIMD operations to our code and see if it gives better performance. It's unlikely, but sometimes [compilers get weird](https://x.com/rflaherty71/status/1894971059885219855) about how they translate high level code into assembly, so we never know for sure what we're going to get. It's worth a shot!

## Optimization #3: x86 Intrinsics

Ok, now we're going to try rewriting our function using the x86 intrinsics provided by the [Rust arch module](https://doc.rust-lang.org/core/arch/index.html). Essentially, we're going to tell the compiler exactly which CPU instruction we want to use for our SIMD operations. This code is going to be even more verbose than the last version, but let's take a look at the new function, and then I'll explain what we're looking at


```rust {linenos=inline}
##[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
pub fn b_spline_x86_intrinsics(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    use std::arch::x86_64::*;
    let mut outputs = Vec::with_capacity(inputs.len());
    let num_k0_activations = knots.len() - 1;
    let mut basis_activations = vec![0.0; num_k0_activations];
    for x in inputs {
        let x_splat = unsafe { _mm512_set1_pd(*x) };

        let mut i = 0;
        // SIMD step for the degree-0 basis functions
        while i + SIMD_WIDTH <= num_k0_activations {
            unsafe {
                let knots_i_vec = _mm512_loadu_pd(&knots[i]);
                let knots_i_plus_1_vec = _mm512_loadu_pd(&knots[i + 1]);
                let left_mask = _mm512_cmp_pd_mask(knots_i_vec, x_splat, _CMP_LE_OQ);
                let right_mask = _mm512_cmp_pd_mask(x_splat, knots_i_plus_1_vec, _CMP_LT_OQ);
                let full_mask = left_mask & right_mask;
                let activation_vec =
                    _mm512_mask_blend_pd(full_mask, _mm512_set1_pd(0.0), _mm512_set1_pd(1.0));

                _mm512_storeu_pd(&mut basis_activations[i], activation_vec);
            }
            i += SIMD_WIDTH;
        }
        // scalar step for the degree-0 basis functions
        while i < num_k0_activations {
            if knots[i] <= *x && *x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
            i += 1;
        }
        for k in 1..=degree {
            let mut i = 0;
            // SIMD step for the higher degree basis functions
            while i + SIMD_WIDTH <= num_k0_activations - k {
                unsafe {
                    let knots_i_vec = _mm512_loadu_pd(&knots[i]);
                    let knots_i_plus_1_vec = _mm512_loadu_pd(&knots[i + 1]);
                    let knots_i_plus_k_vec = _mm512_loadu_pd(&knots[i + k]);
                    let knots_i_plus_k_plus_1_vec = _mm512_loadu_pd(&knots[i + k + 1]);

                    let left_coefficient_numerator_vec = _mm512_sub_pd(x_splat, knots_i_vec);
                    let left_coefficient_denominator_vec =
                        _mm512_sub_pd(knots_i_plus_k_vec, knots_i_vec);
                    let left_coefficient_vec = _mm512_div_pd(
                        left_coefficient_numerator_vec,
                        left_coefficient_denominator_vec,
                    );
                    let left_recursion_vec = _mm512_loadu_pd(&basis_activations[i]);

                    let right_coefficient_numerator_vec =
                        _mm512_sub_pd(knots_i_plus_k_plus_1_vec, x_splat);
                    let right_coefficient_denominator_vec =
                        _mm512_sub_pd(knots_i_plus_k_plus_1_vec, knots_i_plus_1_vec);
                    let right_coefficient_vec = _mm512_div_pd(
                        right_coefficient_numerator_vec,
                        right_coefficient_denominator_vec,
                    );
                    let right_recursion_vec = _mm512_loadu_pd(&basis_activations[i + 1]);

                    let left_val_vec = _mm512_mul_pd(left_coefficient_vec, left_recursion_vec);
                    let right_val_vec = _mm512_mul_pd(right_coefficient_vec, right_recursion_vec);

                    let new_basis_activations_vec = _mm512_add_pd(left_val_vec, right_val_vec);
                    _mm512_storeu_pd(&mut basis_activations[i], new_basis_activations_vec);
                }
                i += SIMD_WIDTH;
            }
            // scalar step for the higher degree basis functions
            while i < num_k0_activations - k {
                let left_coefficient = (x - knots[i]) / (knots[i + k] - knots[i]);
                let left_recursion = basis_activations[i];

                let right_coefficient = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
                let right_recursion = basis_activations[i + 1];

                basis_activations[i] =
                    left_coefficient * left_recursion + right_coefficient * right_recursion;
                i += 1;
            }
        }

        // SIMD step for the final result
        let mut i = 0;
        let mut result = 0.0;
        while i + SIMD_WIDTH <= control_points.len() {
            unsafe {
                let control_points_vec = _mm512_loadu_pd(&control_points[i]);
                let basis_activations_vec = _mm512_loadu_pd(&basis_activations[i]);
                let result_vec = _mm512_mul_pd(control_points_vec, basis_activations_vec);
                result += _mm512_reduce_add_pd(result_vec);
            }
            i += SIMD_WIDTH;
        }
        while i < control_points.len() {
            result += control_points[i] * basis_activations[i];
            i += 1;
        }
        outputs.push(result);
    }

    return outputs;
}
```

Our code gets even more verbose since we can't use pretty things like `a + b` or `a >= b` when working with our data anymore and instead have to use ugly function calls like `_mm512_add_pd(a, b)` and `_mm512_ge_pd(a, b)`

All those `__mm512_*` functions are our x86 intrinsics; They each map directly to a specific assembly instruction provided by the AVX-512 feature. I've heard folks say one of the drawbacks to programming with Intel x86 intrinsics is how hard it makes it to read the code, but frankly I think it's pretty easy once you learn to break it down. Here's how to read the function calls:
* `_mm512` at the front means we're using 512-bit registers - the `ZMM` registers we mentioned before. If we wanted to with `YMM` or `XMM` registers we'd replace the `512` with `256` or `128` respectively
* the `pd` at the end is short for "Precision Double", aka a 64-bit floating-point value. We'd replace this with `epi64` if we wanted to work on 64 bit integers instead of floats
* everything between those two pieces describes the specific operation. `add` means we're adding, `mul` means we're multiplying, `storeu` and `loadu`  means we're storing and loading (and not worrying if the memory address is 64-byte aligned, hence the "u" for "unaligned"), etc.
You can browse through the available operations and their short descriptions on [Rust's doc page for x86_64 intrinsics](https://doc.rust-lang.org/core/arch/x86_64/index.html) or go straight to the source with [Intel's Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=4789)


You may have noticed that all of our intrinsics calls live inside of `unsafe` blocks. The Rust compiler goes to [great pains](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) to ensure any code you write is "safe" and doesn't do things like access invalid memory, modify data shared between threads, or other things that can lead to "undefined behavior" and result in a [significant source of security vulnerabilities in software](https://www.keysight.com/blogs/en/tech/nwvs/2024/03/21/the-impact-of-rust-on-security-development). 

Since our intrinsics calls compile straight to hand-picked assembly instructions, we prevent the compiler from doing things like bounds checks on our loads, or even such basic safety checks as *ensuring valid assembly for the target CPU* (what if we're compiling for an x86 cpu without AVX-512 or even, \*gulp\* an ARM CPU?!). Wrapping the intrinsics code in an `unsafe` block is sort of like signing a liability waiver: we acknowledge that Rust is not providing its normal safety guarantees to code within the block, and it's incumbent on us, the programmer, to ensure our code is "safe". 

We avoid out-of-bounds memory accesses in our SIMD loops by ensuring `i` never refers to an element less than `SIMD_WIDTH` (which equals 8, as defined above) elements from the end, so when we pull 8 64-bit elements with `_mm512_loadu_pd` or write 8 elements with `_m512_storeu_pd`, we're never going past the end of our vectors [^1].

[^1]: That's the Rust Vector type - analogous to a python list or Java ArrayList - not a SIMD vector

And we make sure we actually *have* AVX-512 functionality available with that `#[cfg(...)]` block above our function. That annotation tells the compiler to only include and compile this function when it's compiling for a CPU with the 64-bit x86 architecture and the AVX-512 feature. The AVX-512 requirement is probably sufficient, but we'll include the x86 requirement anyway - a bit of extra safety, with the added bonus of telling any future readers of our code who may not be familiar with AVX-512 that it's related to the x86 architecture. We include the same thing above our benchmark, so the benchmark only gets included when we're building for the proper machine


```rust {linenos=inline}
##[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
##[bench]
fn bench_intrinsic_simd_method(b: &mut Bencher) {
    let (degree, control_points, knots, inputs) = get_test_parameters();
    b.iter(|| {
        let _ = b_spline_x86_intrinsics(&inputs, &control_points, &knots, degree);
    });
}
```


So, that's what it takes to use x86 intrinsics in Rust. What sort of performance improvement does it bring?

```zsh
$> RUSTFLAGS="-Ctarget-cpu=native" cargo bench

...

running 4 tests
test bench_intrinsic_simd_method ... bench:      52,007.98 ns/iter (+/- 1,239.71)
test bench_portable_simd_method  ... bench:      54,111.01 ns/iter (+/- 1,084.46)
test bench_recursive_method      ... bench:     773,143.01 ns/iter (+/- 20,809.34)
test bench_simple_loop_method    ... bench:      63,524.26 ns/iter (+/- 2,502.52)

test result: ok. 0 passed; 0 failed; 0 ignored; 4 measured; 0 filtered out; finished in 15.42s
```

For all the extra verbosity, we got a 1.04x speedup over our portable SIMD code. Is it worth it? There are certainly circumstances where even a 4% reduction in runtime brings sizable benefits, but for a function that can't even run on some processors, I'd hoped for more. Okedoke, let's wrap this thing up...


## Conclusion

We've gone over the concept of SIMD operations and how they're helpful, explained B-Splines and shown how they can be calculated in Rust, and gone over a number of different ways to improve the performance of calculating B-Splines with SIMD operations. Where does that leave us?

![Graph showing relative speeds of our different implementation methods](generated_images/final_times.png)

We get the vast majority of our speedup just moving from a recursive implementation to a loop-based one, getting rid of the recursive overhead and letting LLVM's auto-vectorizer do more optimization work for us; Even if we cut the advantage in half (to account for the fact that the recursive version has to calculate all basis functions besides the top layer twice), it's still a 6x speed boost over recursion. **If you're concerned about performance, stay away from recursion, and embrace loops**.

Rewriting our loop-based method using Rust's portable SIMD module resulted in a non-trivial ~1.15x speedup, but did cost us a bit in code-brevity: we went from 30 lines of code to 300. Now, even though the Rust code tripled, that doesn't mean the compiled assembly necessarily tripled; if it did, we'd be worried about larger functions losing their gains to instruction cache misses. So, even for larger functions, **moving to a portable SIMD implementation may be worth it for performance-sensitive programs**.

We'd hoped that by rewriting our function using explicit SIMD code, we'd be able to take advantage of the bigger 512-bit registers and go even faster. disappointingly, while we did in-fact improve data-level parallelism, reducing the total number of instructions required to complete our calculations, those gains were cancelled out by worse instruction-level parallelism and thus fewer instructions-per-cycle, resulting in essentially unchanged runtime. 

That being said, once we'd written the portable SIMD code, it was trivial to enable the 512-bit operations - we just had to pass a flag to our compiler. If you're able to compile and benchmark your code on the same type of machine that will run it, **it's worth it to pass `-Ctarget-cpu=native` to the compiler and see what happens**

Finally, using intrinsics to inject SIMD operations into our code had a disappointing return on investment. At a minor cost of readability and a significant cost of *portability*, we only eked out a measly 1.04x speedup. Those extremely concerned with performance may cheer *anything* that could improve speed by 4%, but for everyone else, the cost of maintaining a separate portable version of the optimized functions and conditionally compile the correct version will outweigh the cost. **Using SIMD intrinsics only provides a marginal speed-up over Portable SIMD.** 

| Implementation     | Speedup over Recursive | Speedup over Next-Best | Additional Requirements                                         |
| ------------------ | ---------------------- | ---------------------- | --------------------------------------------------------------- |
| Recursive          | 1.0x                   | 1.0x                   | None                                                            |
| Loop-based         | 12.2x                  | 12.2x                  | None                                                            |
| Portable SIMD      | 14.3x                  | 1.17x                  | nightly Rust                                                    |
| CPU Intrinsic SIMD | 14.9x                  | 1.04x                  | nightly Rust + separate implementation for different processors |

Just because you're not running on a GPU doesn't mean you can't take advantage of greater parallelism, but nothing is free. SIMD operations are an option when your CPU supports it, but I would only recommend it for the most speed-conscious; for everyone else, focusing on generally-efficient code will provide the most bang-for-your-buck