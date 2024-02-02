using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random
using BSON

#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 17662659248  ## <---replace 0 by your student_number 
### ---- ###
## In this tiny HW you will implement gradient descent with nesterov momentum
## You can use the corresponding jl files for reference.... see github week 11.
## You will only need to implement optimize function all the rest is given to you...

abstract type AbstractOptimiser end
mutable struct Nesterov <: AbstractOptimiser
    eta::Float64
    rho::Float64
    velocity::IdDict
end
  

Nesterov(η = 0.001, ρ = 0.99) = Nesterov(η, ρ, IdDict())

## You need to implement apply! function, for this you will need to review: Week 11 
function apply!(o::Nesterov, x, Δ)
    η, ρ = o.eta, o.rho
    v = get!(() -> zero(x), o.velocity, x)::typeof(x)
    @. v = ρ*v - η*Δ
    @. Δ = -v
end

## Step function is given, because it is the same...
function step!(opt::AbstractOptimiser, x::AbstractArray, Δ)
    x .-= apply!(opt,x, Δ)
    return x
end 

##You will now implement optimize function, remember that you will take the gradient at point where you will use
### velocity factor, therefore may wish to use get function...
function optimize(f::Function, x::AbstractArray, opt::AbstractOptimiser; max_iter = 2, stopping_criterion = 1e-10)
    for i in 1:max_iter
        grad = Zygote.gradient(t->f(t), x)[1]
        x = step!(opt, x, grad)
        if norm(grad) < stopping_criterion
            @info "ok in $(i) steps"
            return x
        end
    end
    @info "No convergence buddy!!!"
    return x
end



function unit_test()
    opt = Nesterov(0.001, 0.99)
    x = begin
        Random.seed!(0)
        randn(2)
    end
    g(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1]+x[1]*x[2]^2)^2 + (2.625 -x[1]+ x[1]*x[2]^3)^2
    x_ = optimize(t->g(t), x, opt; max_iter = 100000)
    println("Things seem to work!!! The point real point shold be [3.0, 0.5] while yourse is $(x_)")
end

## Let's see what you doin' 
unit_test()
##

# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
   println("Ok Computer!!!")
end
