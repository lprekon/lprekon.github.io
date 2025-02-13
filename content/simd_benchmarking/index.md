+++
date = '2025-02-12T19:00:59-05:00'
draft = true
title = 'Index'
+++
# Benchmarking Different Vectorization Strategies in Rust

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
lorem ipsum
### sliding window method
lorem ipsum
### pyramid method
lorem ipsum