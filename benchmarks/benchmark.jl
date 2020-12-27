using BenchmarkTools
using Ramnet
using MLDatasets

# Should I make this a standard matrix? I may be benchmarking the transpose operation if it's lazy
X_train = permutedims(reshape(MNIST.traintensor(), 784, :)) .> 0.5
y_train = MNIST.trainlabels()

# b_train = @benchmarkable train!(model, $X_train, $y_train) setup = (model = MultiDiscriminatorClassifier{Int64}(784, 28; seed=1))

# display(run(b_train))

# println("\n2) MultiDiscriminatorClassifier inference with the MNIST dataset(10.000 test images):")

# model = MultiDiscriminatorClassifier{Int64}(784, 28; seed=1)
# train!(model, X_train, y_train)

X_test = permutedims(reshape(MNIST.testtensor(), 784, :)) .> 0.5

# b_predict = @benchmarkable predict($model, $X_test)

# display(run(b_predict))

# println()

########

const suite = BenchmarkGroup()

# Subgroups
suite["mapper"] = BenchmarkGroup(["mappers", "mapping"])
suite["mapper"]["random mapper"] = BenchmarkGroup(["random mapping"])
suite["node"] = BenchmarkGroup()
suite["node"]["dict node"] = BenchmarkGroup()
suite["discriminator"] = BenchmarkGroup(["discriminators"])

# Random mapping tests
suite["mapper"]["random mapper"]["Instantiation"] = @benchmarkable RandomMapper(784, 28; seed=1)

mapper = RandomMapper(784, 28; seed=1)
mapper_itr = map(mapper, X_train)

suite["mapper"]["random mapper"]["Iteration"] = @benchmarkable collect(map($mapper, $X_train))
# suite["mapper"]["random mapper"]["Iteration"] = @benchmarkable begin
#     for (i, tuple) in Iterators.enumerate(map($mapper, $X_train))
#         tuples[i] = tuple
#     end
# end setup = (tuples = Array{eltype(mapper_itr)}(undef, length(mapper_itr)))

# Node tests
zero_digits = X_train[findall(y_train .== 0), :]
tuple = first(map(RandomMapper(784, 28; seed=1), zero_digits))

suite["node"]["dict node"]["train (Vector{Bool})"] = @benchmarkable Ramnet.Nodes.train!(node, tuple) setup = (node = Ramnet.Nodes.DictNode{Vector{Bool}}())
suite["node"]["dict node"]["train (BitVector)"] = @benchmarkable Ramnet.Nodes.train!(node, tuple) setup = (node = Ramnet.Nodes.DictNode{BitVector}())

paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(suite)
    BenchmarkTools.save(paramspath, params(suite));
end