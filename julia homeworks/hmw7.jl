using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random
using Flux
using MLUtils
using NNlib  
using Random
using PyCall 


#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 17662659248  ## <---replace 0 by your student_number 
### ---- ###
## In this HW you are on your own, 

## Please change the path of the london_weather.csv which I use for this homework. 
df = CSV.read("/Users/ummuhanyirmili/Desktop/london_weather.csv", DataFrame)

dropmissing!(df, :mean_temp)

# Create a binary target variable: 1 for 'hot' and 0 for 'cold'
threshold = median(df.mean_temp)
df.is_hot = Int.(df.mean_temp .> threshold)

# Drop the 'date' and 'mean_temp' columns
select!(df, Not([:date, :mean_temp]))

# Convert to arrays
X = Matrix(df[:, 1:end-1])
y = Vector(df[:, end])


train_indices = shuffle(1:size(X, 1))
train_size = floor(Int, size(X, 1) * 0.8)
X_train = X[train_indices[1:train_size], :]
y_train = y[train_indices[1:train_size]]
X_test = X[train_indices[train_size+1:end], :]
y_test = y[train_indices[train_size+1:end]]

model = Chain(
    Dense(size(X_train, 2), 64, relu),
    Dense(64, 8, σ)
)

loss(x, y) = Flux.binarycrossentropy(model(x), y)
optimizer = ADAM(0.001)

# parameter
model_params = Flux.params(model)
epochs=100

##Training model
for epoch in 1:epochs
    local train_loss
   
    for (x, y) in DataLoader(eachrow(Array(X_train)), batchsize=10, shuffle=true)
        gs = Flux.gradient(model_params) do
            return loss(x, y)
        end
        Flux.update!(optimizer, model_params, gs)
        train_loss = loss(x, y)
        if epoch % train_loss_step == 0
            @printf("Epoch: %d, Loss: %.4f\n", epoch, train_loss)
        end
    end

    if accuracy(Array(X_train), Array(y_train)) >= accuracy_threshold
        @printf("Epoch: %d, Accuracy Threshold Met. Stopping Training.\n", epoch)
        break
    end
end

# For model evaluation
accuracy(x, y) = mean(Flux.σ.(model(x)) .> 0.5 .== y)
println("Test Accuracy: $(accuracy(X_test', y_test))")


# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
    @assert student_number != 0
    println("Seems everything is ok!!!")
end
