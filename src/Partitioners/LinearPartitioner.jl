struct LinearPartitioner <: AbstractPartitioner
    width::Int
    n::Int

    function LinearPartitioner(width::Int, n::Int)
        width < 0 && throw(DomainError(width, "Width must be non-negative"))
        n < 1 && throw(DomainError(n, "Partition may not be less then one"))

        new(width, n)
    end
end

function Base.iterate(itr::PartitionerIterator{LinearPartitioner,<:AbstractVector}, state=1)
    l = length(itr)
    start = (state - 1) * itr.partitioner.n + 1

    if state > l
        return nothing
    elseif state == l
        return itr.X[start:end], state + 1
    else
        return itr.X[start:start + itr.partitioner.n - 1], state + 1
    end
end

function Base.iterate(itr::PartitionerIterator{LinearPartitioner,<:AbstractMatrix}, state=1)
    l = length(itr)
    start = (state - 1) * itr.partitioner.n + 1

    if state > l
        return nothing
    elseif state == l
        return itr.X[:, start:end], state + 1
    else
        return itr.X[:, start:start + itr.partitioner.n - 1], state + 1
    end
end