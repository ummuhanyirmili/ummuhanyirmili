## Dear Comrades,
## You shall now implement bisection method for finding minima of a fucntion

## In this HW you have two tasks
## 1) Implement numerical derivative (see https://en.wikipedia.org/wiki/Numerical_differentiation)
## 2) Use this numerica derivative in bisection method (see the slides on the main page for details)
## 3) Let this song help you https://www.youtube.com/watch?v=D3-0qh6qHfU


### Run the following part to see if there is any package that you need to install
### if something goes wrong, watch the error message and install the needed package by 
### Pkg.add("missing_package")
using Pkg
using BSON
import Base: isapprox
### Run the following function as it is needed for comparison
function isapprox((a,b)::Tuple{T, T}, (c,d)::Tuple{L, L}; rtol = 1e-2) where {T <: Real, L <: Real}
    return abs(a-c) < rtol && abs(b-d) < rtol
end

###
###  
### Scroll down for your HW
### Before getting started you should write your student_number in integer format
const student_number::Int64 = 17662659248 ## <---replace 0 by your student_number 
###
###




###### Here we start buddy lean back and have a cup of coffee--- make sure that the song is playing in the background!!! #######


#### Below you will implement approximate derivative function ####
### the following function, you shall approximately find its derivatives as follows
function derivative(f::Function, x::Real; h::Float64 = 1e-5)
    return (f(x+h) - f(x))/h
end


#### What is the derivative of f(x) = x^10 at x = 1? You wanna confirm??
derivative(x->x^10, 1)
### Goodd job!!!

function unit_test_for_derivative()
    @assert !(student_number == 0) "Write your student number"
    @assert isa(student_number, Int64) "Your student number should be in integer format"
    @assert isa(derivative(x->x^2, 1), Float64) "Return type should be a Float64"
    try
        @assert isapprox(derivative(x->x^2, 1.0), 2.0; rtol = 1e-3)
        @assert isapprox(derivative(x->sin(x), 0.0), 1.0; rtol = 1e-3)
        @assert isapprox(derivative(x->exp(x), 10), exp(10); rtol = 1e-3)                
    catch AssertionError
        @info "Something went wrong buddy. Checkout your implementation"
        throw(AssertionError)
    end
    @info "In Tonny Montana style: things are fine now!!!! OKKAYYY great successs"
    return 1
end

###  --- Run Here to see if your function does a good job --- ###
unit_test_for_derivative()
###  --- end of unit test for derivative function ---- ###


#= You will now implement bisection method before you get started please see the slides for a detailed information!!!!
    Here ϵ stands for stopping criteria!!!, your function should return a tuple of the form α, f(α) 
    where α is the point where your function is minimized and f(α) is the value of your function at this point!!!!
    please watchaaa the argument types !!!!!!!!!
=#
function find_minimum_bisection(f::Function, a::Real, b::Real; max_iter::Int = 100, ϵ::Float64 = 1e-5)::Tuple{Float64, Float64}
    @assert a<b "a should be less than b"
    ### Your code below
    for i in 1:max_iter
        α = (a + b) / 2.0  # midpoint
        derivative_at_α = derivative(f, α)
        
        if derivative_at_α < 0
            a = α
        else
            b = α
        end
        
        if abs(a - b) < ϵ
            @info "Converged after $i iterations"
            return α, f(α)
        end
    end
    ### your code above
    
    ## in the case that ϵ criteria is not met!!! return the following!!!
    @info "Algorithm did not converge correctly!!!"
    return α, f(α)
end

## Before going to unit_test run the next cell see what ya doin?
find_minimum_bisection(x->x^2-1, 0.0, pi)
### 

## Unit test for bisection ###
function unit_test_for_bisection()
    try
       @assert isapprox(find_minimum_bisection(x->x^2 - 1, -1, 1), (0.0, -1.0); rtol = 1e-3)
       @assert isapprox(find_minimum_bisection(x->-sin(x), 0.0, pi), (pi/2, -1.0); rtol = 1e-3)
       @assert isapprox(find_minimum_bisection(x->x^4+x^2+x, -1, 1), (-0.3859, -0.2148); rtol = 1e-3) 
    catch AssertionError
        @info "Something went wrong buddy checkout your implementation"
        throw(AssertionError)
    end
    @info "In Tonny Montana style: things are fine now!!!! OKKAYYY"
    return 1
end   


## Run the unit_test_for_bisection to see if your doing goood!!!
unit_test_for_bisection()
###

#### Seems that we are done here. Congrats comrade, you have completed this task successsssfuly.
####
####
####

####
####
####

####
####
####

####
####
####


###### do not change anything below!!!!
##### No need to run anything below!!!!
###### As this may cause compiler to crash, and degradation!!!
if abspath(PROGRAM_FILE) == @__FILE__

    G::Int64 = unit_test_for_bisection()  + unit_test_for_derivative()
    dict_ = Dict("std_ID"=>student_number, "G"=>G)
    try
        BSON.@save "$(student_number).res" dict_ 
    catch Exception 
            println("something went wrong with", Exception)
    end
end
