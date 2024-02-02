
### Run the following part to see if there is any package that you need to install
### if something goes wrong, watch the error message and install the needed package by 
### Pkg.add("missing_package")
using Pkg
using Plots: plot, plot!
using Statistics
using BSON
using Random
using LinearAlgebra
using Zygote


#### ----- ###
cd(@__DIR__)
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 17662659248  ## <---replace 0 by your student_number 
### ---- ###
## Assume that you are given a matrix A and column vector b, 
## you want to determine x so that Ax is close to b as much as possible use LinearAlgebra.norm!!!!.
## and solve this problem as an unconstrained optimization problem.
## We will now do this step by step. 
## Given A and b,
## 1) Implement your objective function 
## 2) Fix your learning rate, and max_iter
## 3) Do gradient descent until some convergence stopping_criterions are met!!!

## Step 1) Implement your objective function
function objective(A::AbstractMatrix, b::AbstractVector, x::AbstractVector)::AbstractFloat

    residual = A * x - b

    norm = LinearAlgebra.norm(residual)
     
    return norm 
end

### Some unit test as usual!!!!
function unit_test_objective()::Int64
    @assert student_number != 49 "Watchout your student number pal!!!"
    for _ in 1:100
        let 
            A,b,x = randn(100, 100), randn(100), randn(100)
            N = (transpose(A*x-b)*(A*x-b) |> sum |> sqrt)
            @assert isapprox(objective(A,b,x),N) "Something went wrong!!!"
        end
    end
    @info "Oki Doki!!!"
    return 1
end

unit_test_objective()

## 2) We will do gradient descent explicitly that is we give the initial points
## and the algorithm will give us back probably the local minimizer!!!
## the following function should return a tuple of local minimum and the gradient of the objective function
## Please mind the order !!!!
## Your for loop should stop either when you reach max_iter or when the gradient is smaller than
## No println functions should be implemented, no @info macros should be implemented!!!
## stoppin criterion.

function find_minimum(loss::Function, 
    x_init::AbstractVector; 
    learning_rate::Float64,
    max_iter::Int64, 
    stopping_criterion::Float64 = 1e-3)::Tuple{Vector{Float64}, Vector{Float64}}
x = copy(x_init)
grad = gradient(loss, x)[1]

for i in 1:max_iter

x -= learning_rate * grad


grad = gradient(loss, x)[1]

if norm(grad) < stopping_criterion
    break
end

end

return (x, grad)
end


## Let's give a try before jumping to unit tests!!!!!
find_minimum(x->x[1]^2 + x[2]^2, 5*randn(2), learning_rate = 0.1, max_iter = 1000, stopping_criterion = 1e-20)
## Oki doki if you see something very close to zero, zero


function unit_test_minimum()::Int64
    Random.seed!(0)
    @assert isa(find_minimum(x->x[1]^2 + x[2]^2, randn(2);learning_rate = 0.1, max_iter = 1000), Tuple{Vector{T}, Vector{T}} where T<:Real) "The return type should be Tuple{Float64, Float64}"
    @assert find_minimum(x->x[1]^2 + x[2]^2, randn(2); stopping_criterion = 1e-5, learning_rate = 0.1, max_iter = 1000)[2] |> norm < 1e-5 "Check out the stopping criterion!!!"
    @assert isapprox(find_minimum(x->x[1]^2 + x[2]^2, randn(2); learning_rate = 0.01, max_iter = 1000)[1], zeros(2); atol = 0.5) "Your algo does not work!!!"
    @info "Basic unit tests are OK! Let's now do some more additional tests..., it may take some time BTW, while we perform the tests, you can enjoy https://www.youtube.com/watch?v=bn1YCClRF-g"
    let 
        Random.seed!(0)
        A = randn(10,10)
        b = randn(10)
        min = find_minimum(x->objective(A,b,x), randn(10), learning_rate = 0.01, max_iter = 1000000)[1]
        @assert isapprox(min, A\b, atol = 0.5) "Oh no... oh no... oh no oh no oh no  :....("
        @info "CongratSSSSSSS have some rest!!!!"
    end
    return 1
end


## Run the next function to see you are doing good!!!
unit_test_minimum()
##Great!!!!

## No need to run below!!!
if abspath(PROGRAM_FILE) == @__FILE__
    G::Int64 = unit_test_objective()+unit_test_minimum()
    dict_ = Dict("std_ID"=>student_number, "G"=>G)
    try
        BSON.@save "$(student_number).res" dict_ 
        catch Exception 
            println("something went wrong with", Exception)
    end

end

