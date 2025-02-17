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

Now we're cooking with gas! Here we see a B-spline in all it's glory. By manipulating the control points, we can "tug" portions of the spline curve in one direction or another. 

Through the proper choice of knots, control points, and degree, we can use B-Splines to construct arbitrary curves, and thus approximate any function we want

There's a lot more we could say about B-Splines (what happens if we mess with the knots? How what difference does increasing the degree make? How do B-Splines work in 2 or more dimensions?), but that's beyond the scope of this post. For those interested, see: 
* [Shape Interrogation for Computer Aided Design and Manufacturing, Chapter 1.4](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node15.html), MIT Hyperbook
* [Definition of a B-Spline Curve](https://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/B-spline.pdf), UC Lecture notes
* [Desmos B-Spline Playground](https://www.desmos.com/calculator/ql6jqgdabs)
* and of course [B-Spline](https://en.wikipedia.org/wiki/B-spline), Wikipedia

**In conclusion: B-Splines are functions that let us trace arbitrary curves. To determine the value of the spline at some point `x`:**
1. **evaluate each basis function (which is a recursive function) at `x`**
2. **multiply by the basis function outputs by their corresponding conrtrol points**
3. **sum the results**

Now that that's out of the way, let's take a look at the code we'll be optimizing

### Evaluating a B-Spline with Rust

Let's begin with a straight forward implementation of our spline math. This code is intentionally sub-optimal, so we'll have a baseline for our benchmarking

```rust
/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `x`
fn basis_activation(i: usize, k: usize, x: f64, knots: &[f64]) -> f64 {
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
fn b_spline(x: f64, control_points: &[f64], knots: &[f64], degree: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..control_points.len() {
        result += control_points[i] * basis_activation(i, degree, x, knots);
    }
    return result;
}

```

We'll be benchmarking this code with Rust's built-in benchmarking tool [`cargo bench`](https://doc.rust-lang.org/cargo/commands/cargo-bench.html). Here's a quick look at our benchmarking code

```rust
#![feature(test)]
extern crate test;
use test::Bencher;

use rust_simd_becnhmarking::b_spline;

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

#[bench]
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
>$ cargo bench -q

running 4 tests
iiii
test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 1 test
test bench_recursive_method ... bench:     706,262.10 ns/iter (+/- 81,587.11)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured; 0 filtered out; finished in 0.22s
```

Looks like evaluating all 100 inputs takes about 800k nanoseconds, or roughly 0.8 milliseconds. Cool, we have a baseline. Now we can start optimizing

From this point on, as we explore different vectorization strategies, there are lots of little details that a competant programmer might overlook at first, but which can have a large impact on performance - ineffecient memory allocation, redundant looping, etc. I went through rigamaroll of write-profile-optimize when I first wrote these operations as part of my [KAN Library](https://crates.io/crates/fekan). I will quietly include the results of all those lessons-learned in the code I show going forward, because I want each iteration of the algorithm to be the best version of itself it can be.

## Optimization #1: From Recursion to Looping

To understand our first optimization, let's take a step back and consider how the value of `B_i_k`, the value of the `i'th` basis function of degree `k`, depends on the values of the basis functions of degree `k-1`

![A pyramid showing the dependency chain for the 0th basis function at degree 3](generated_images/single_basis_pyramid.png)

One thing to note is that in each layer, each basis function is depended on by the one above it, and the one above-and-to-the-left of it. Our actual B-splines depend on more than one top-level basis function, however, so let's look at a version of this pyramid with multiple basis functions in the top layer

![A pyramid showing the dependency chain for the 0th through 3rd basis function at degree 3](generated_images/multiple_basis_pyramid.png)

A basis function is never depended **on** by *any* basis function to its right, and a basis function never **depends** on *any* basis function to its left. With that, we can rewrite our spline function in a loop that reuses previously calculated values instead of throwing them away.

```rust
/// Calculate the value of the B-spline at the given parameter `x` by looping over the basis functions
pub fn b_spline_loop_over_basis(
    inputs: &[f64],
    control_points: &[f64],
    knots: &[f64],
    degree: usize,
) -> Vec<f64> {
    let mut outputs = Vec::with_capacity(inputs.len());
    let mut basis_activations = vec![0.0; knots.len() - 1];
    // fill the basis activations vec with the valued of the degree-0 basis functions
    for x in inputs{
        let x = *x; 
        for i in 0..knots.len() - 1 {
            if knots[i] <= x && x < knots[i + 1] {
                basis_activations[i] = 1.0;
            } else {
                basis_activations[i] = 0.0;
            }
        }

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

        let mut result = 0.0;
        for i in 0..control_points.len() {
            result += control_points[i] * basis_activations[i];
        }
        outputs.push(result);
    }
    return outputs;
}
```

Here's our first optimized spline calculator. Now that we're not recursing, there's no need for a separate basis function - we do all our calculations in this one spline function. We're also taking in a whole batch of input values to be processed at once, instead of only taking one at a time, for efficieny reasons that will be explained in a moment.

In order to calculate the final value of the spline at point `x`, we need the value of each of our top level basis functions. To start, in that `0..knots.len() - 1` loop, we calculate the value of each degree-0 basis functions and store the results in a vector. 

![degree-3 pyramid of basis functions with all but the bottom layer greyed out](generated_images/basis_pyramid/multiple_basis_pyramid_bot_layer_filled.png)

Next, the `1..=degree` loop is where the magic happens. At each layer `k`, starting at `1` and moving up to our full degree, we walk our vector of basis functions and calculate each in turn, overwriting the value of the lower-degree basis function that was in its spot. This works because of the direction of the arrows in the dependency pyramid. For example, when `k=1` and `i=0`, we're calculating basis function `B_0_1`, which depends on `B_0_0` and `B_1_0`, which at that point live in our vector at the `0th` and `1st` position, respectively. We read those values from the vector, and use them to calculate `B_0_1`

![degree-3 pyramid of basis functions with all but the bottom layer greyed out. There's a red box around the first basis function in the second layer](generated_images/basis_pyramid/calculating_B_0_3.png)

Then we write `B_0_1` to the `0th` position in our vector, overwritting `B_0_0`, which is no longer needed. After that we move on to calculating `B_1_1`

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
1. Reduced overhead. Besides the work done within a function, there's a certain amount of work required simply to call a functiona and return from it. Our recursive method had a lot of function calls - now we have only one
2. Reusing calculated basis values. Go back and look at our pyramid of basis functions; each basis function `B_i_k` is depended on by two other basis functions - `B_i-1_k+1` and `B_i_k+1`. In the recursive method, we'd calculate `B_i_k` once while calculating its first dependent, and again when calculating its second dependent. Now that we're storing the results of each basis function calculation in our vector, we only need to calculate each one once
3. **Auto-vectorization**. In recursive mode, the compiler was limited in what it could assume about our code, so it was forced to be conservative in how it optimizied and wrote assembly to do exactly what we described and nothing more - read a few values, multiply and add them together, and give a single value back. Now that we're working a loop, the compiler is able to recognize that we're walking a vector and performing the same operation at each step, and do things smarter: the compiler is generating assembly with SIMD operations. While our Rust code says "for each index 0..n, read a few values, multiply and add them together, and store the single result", the assembly generated by the compiler now says "for every chunk of indexes [0..i]...[n-i..n], read several chunks of values, multiply and add the chunks together, and store the several results all at once". We're getting vectorization for free, just by writing code that's easier for the compiler to understand!


## Optimization #2: Rust's Portable SIMD Crate

Now we'll start introducing SIMD operations using Rust's [portable SIMD module](https://doc.rust-lang.org/std/simd/index.html). 

Note for those following along with their own code at home: using `std::simd` requires addings the `#![feature(portable_simd)]` flag at the top of our library and compiling with the nightly toolchain, instead of the default stable release. You can install the `nightly` toolchain using [rustup] (https://www.rust-lang.org/tools/install) with `rustup toolchain install nightly`, and set it as the default toolchain for your project by calling `rustup override set nightly` from within your project directory

Below is our B-spline calculation function using SIMD operations. It calculates everything the same way as our looping method, but uses explicit SIMD calls to operate on multiple elements at the same time

```rust
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
[ec2-user@ip-172-31-22-254 spline_simd_benchmarking]$ cargo bench -q

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

There are a lot of tools available to inspect assembly - for this investigation I used [ghidra](https://ghidra-sre.org). Let's see what our `k in 1..=degree` loop looks like once it's compiled

![The k>=1 basis calculation loop, compiled with `cargo build -r`](images/portable_first_look.png)

On the left we have the main workhorse loop of our spline calculations, and on the right is a portion of the assembly code for that loop. One thing jumps out immediately: **we're only using 128-bit SIMD operations, instead of the expected 512-bit**. In [x86 assembly SIMD operations](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), the `XMM` mneumonic is used to refer to 128-bit registers; `YMM` refers to 256-bit registers, and `ZMM` refers to 512-bit registers. 

In our code, we set the constant `SIMD_WIDTH = 8`, which is then passed to the rust simd code to control how many values get packed together. Since our code says to pack together 8 64-bit values, and 8x64=512, we'd expect to see `ZMM` littered throughout our assembly, but it's missing. Since we see `XMM` throughout, we can deduce that the code is only using 128-bit operations

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

Despite the fact that an [Intel 4th gen Xeon](https://aws.amazon.com/ec2/instance-types/c7i/) processor, which [absolutely has](https://en.wikipedia.org/wiki/Sapphire_Rapids) the AVX-512 feature (and thus 512-bit capabilities), the compiler is targeting a generic x86 CPU, and believes it can only use up to SSE and SSE2 feature sets (which explains the `XMM` registers we saw in the assembly code). In order to use the full feature set of our processor, we need to tell the compiler specifically what sort of processer it ought to compile for. We do this with the [`target-cpu` flag](https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu). Let's ask the compiler what features it thinks our Sapphire Rapids CPU has.

```zsh
[ec2-user@ip-172-31-22-254 spline_simd_benchmarking]$ rustc --print cfg -Ctarget-cpu=sapphirerapids | grep feature
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

The compiler knows that a Sapphire Rapids CPU can handle the full range of AVX-512 operations, so we just need to tell the compiler that it should in fact compile for Sapphire Rapids, by passing `-Ctarget-cpu=sapphirerapids` in when we compiled (you can also use `-Ctarget-cpu=native` to tell the compiler "target whatever CPU you're currently on"). We need pass the flag to the compiler through the `RUSTFLAGS` environment variable since we're calling `cargo` instead of calling `rustc` directly. 

Let's recompile our code and take another look in Ghidra
![the k>=1 basis calculation loop, compiled with `RUSTFLAGS="-C target-cpu=sapphirerapids" cargo build -r`](images/portable_target_cpu.png)

*Now* we see the `ZMM` register usage we expect! We've succesfully convinced the compiler to take full advantage of the 512-bit circuitry present in our CPU. Since we've 4x'd the size of the SIMD operations used by our program (moving from 128-bit `XMM` registers to 512-bit `ZMM` registers), we should expect close to a 4x speedup!

```zsh
[ec2-user@ip-172-31-22-254 spline_simd_benchmarking]$ RUSTFLAGS="-Ctarget-cpu=sapphirerapids" cargo bench -q

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

We got... no speedup? Not only that, it's actually a little bit **slower**. This is quite counter-intuitive, and deserves additional investigation. Let's go through some reasons we might see this performance non-change, and try and rule out as many as we can