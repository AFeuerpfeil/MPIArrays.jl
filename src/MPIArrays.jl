"""
Arrays with fixed size and MPI-safe indexing. This vector type is not meant for allocations in hot loops, but rather a convenience wrapper to use MPI on code without having to rewrite all array accesses. Examples for applications are distributed tensor networks.
To make sure no errors arrise, it is crucial that all distributed operations like Base.setindex! or operations that change the size of the Array are performed on all ranks.

The design of this package is inspired by ["CircularArrays.jl"].
"""
module MPIArrays

export MPIArray, MPIVector, MPIMatrix

using MPI, MPIHelper

"""
    MPIArray{T, N, A} <: AbstractArray{T, N}
`N`-dimensional array backed by an `AbstractArray{T, N}` of type `A` with fixed size and MPI-safe indexing.
"""
struct MPIArray{T, N, A <: AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    MPIArray{T,N}(data::A) where A <: AbstractArray{T,N} where {T,N} = new{T,N,A}(data)
    MPIArray{T,N,A}(data::A) where A <: AbstractArray{T,N} where {T,N} = new{T,N,A}(data)
end

"""
    MPIVector{T,A} <: AbstractVector{T}
One-dimensional array backed by an `AbstractArray{T, 1}` of type `A` with fixed size and MPI-safe indexing.
Alias for [`MPIArray{T,1,A}`](@ref).
"""
const MPIVector{T} = MPIArray{T, 1}

"""
    MPIMatrix{T,A} <: AbstractMatrix{T}
Two-dimensional array backed by an `AbstractArray{T, 2}` of type `A` with fixed size and MPI-safe indexing.
Alias for [`MPIArray{T,2,A}`](@ref).
"""
const MPIMatrix{T} = MPIArray{T, 2}

"""
    MPIArray(data)
Create a `MPIArray` wrapping the array `data`.
"""
MPIArray(data::AbstractArray{T,N}) where {T,N} = MPIArray{T,N}(data)
MPIArray{T}(data::AbstractArray{T,N}) where {T,N} = MPIArray{T,N}(data)

"""
    MPIArray(def, size)
Create a `MPIArray` of size `size` filled with value `def`.
"""
MPIArray(def::T, size) where T = MPIArray(fill(def, size))

Base.IndexStyle(::Type{MPIArray{T,N,A}}) where {T,N,A} = IndexCartesian()
Base.IndexStyle(::Type{<:MPIVector}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(arr::MPIArray, i::Int) = getindex(arr.data, i)
Base.@propagate_inbounds Base.getindex(arr::MPIArray{T,N,A}, I::Vararg{Int,N}) where {T,N,A} = getindex(arr.data, I...)

Base.@propagate_inbounds function Base.setindex!(arr::MPIArray, v, i::Int)
    if MPI.Initialized()
        comm_world = MPI.COMM_WORLD
        if MPI.Comm_rank(comm_world) == 0
            setindex!(arr.data, v, i)
        end
        return arr.data[i] = large_bcast(arr.data[i], comm_world; root=0)
    else
        return setindex!(arr.data, v, i)
    end
end
Base.@propagate_inbounds function Base.setindex!(arr::MPIArray{T,N,A}, v, I::Vararg{Int,N}) where {T,N,A}
    if MPI.Initialized()
        comm_world = MPI.COMM_WORLD
        if MPI.Comm_rank(comm_world) == 0
            setindex!(arr.data, v, I...)
        end
        return arr.data[I...] = large_bcast(arr.data[I...], comm_world; root=0)
    else
        return setindex!(arr.data, v, I...)
    end
end

@inline Base.size(arr::MPIArray) = size(arr.data)
@inline Base.axes(arr::MPIArray) = axes(arr.data)
@inline Base.parent(arr::MPIArray) = arr.data

@inline Base.iterate(arr::MPIArray, i...) = iterate(parent(arr), i...)

@inline Base.in(x, arr::MPIArray) = in(x, parent(arr))
@inline Base.copy(arr::MPIArray) = MPIArray(copy(parent(arr)))

@inline function Base.checkbounds(arr::MPIArray, I...)
    J = Base.to_indices(arr, I)
    length(J) == 1 || length(J) >= ndims(arr) || throw(BoundsError(arr, I))
    nothing
end

@inline _similar(arr::MPIArray, ::Type{T}, dims) where T = MPIArray(similar(parent(arr), T, dims))
@inline Base.similar(arr::MPIArray, ::Type{T}, dims::Tuple{Base.DimOrInd, Vararg{Base.DimOrInd}}) where T = _similar(arr, T, dims)
# Ambiguity resolution with Base
@inline Base.similar(arr::MPIArray, ::Type{T}, dims::Dims) where T = _similar(arr, T, dims)
@inline Base.similar(arr::MPIArray, ::Type{T}, dims::Tuple{Integer, Vararg{Integer}}) where T = _similar(arr, T, dims)
@inline Base.similar(arr::MPIArray, ::Type{T}, dims::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}}) where T = _similar(arr, T, dims)

@inline _similar(::Type{MPIArray{T,N,A}}, dims) where {T,N,A} = MPIArray{T,N}(similar(A, dims))
@inline Base.similar(CA::Type{MPIArray{T,N,A}}, dims::Tuple{Base.DimOrInd, Vararg{Base.DimOrInd}}) where {T,N,A} = _similar(CA, dims)
# Ambiguity resolution with Base
@inline Base.similar(CA::Type{MPIArray{T,N,A}}, dims::Dims) where {T,N,A} = _similar(CA, dims)
@inline Base.similar(CA::Type{MPIArray{T,N,A}}, dims::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}}) where {T,N,A} = _similar(CA, dims)

@inline Broadcast.BroadcastStyle(::Type{MPIArray{T,N,A}}) where {T,N,A} = Broadcast.ArrayStyle{MPIArray{T,N,A}}()
@inline Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MPIArray{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType} = MPIArray(similar(convert(Broadcast.Broadcasted{typeof(Broadcast.BroadcastStyle(A))}, bc), ElType))

@inline Base.dataids(arr::MPIArray) = Base.dataids(parent(arr))

function Base.showarg(io::IO, arr::MPIArray, toplevel)
    print(io, ndims(arr) == 1 ? "MPIVector(" : "MPIArray(")
    Base.showarg(io, parent(arr), false)
    print(io, ')')
    # toplevel && print(io, " with eltype ", eltype(arr))
end

"""
    MPIVector(data)
Create a `MPIVector` wrapping the array `data`.
"""
MPIVector(data::AbstractArray{T, 1}) where T = MPIVector{T}(data)

"""
    MPIMatrix(data)
Create a `MPIMatrix` wrapping the array `data`.
"""
MPIMatrix(data::AbstractArray{T, 2}) where T = MPIMatrix{T}(data)


"""
    MPIVector(def, size)
Create a `MPIVector` of size `size` filled with value `def`.
"""
MPIVector(def::T, size::Int) where T = MPIVector{T}(fill(def, size))

"""
    MPIMatrix(def, size)
Create a `MPIMatrix` of size `size` filled with value `def`.
"""
MPIMatrix(def::T, size::NTuple{2, Integer}) where T = MPIMatrix{T}(fill(def, size))

Base.empty(::MPIVector{T}, ::Type{U}=T) where {T,U} = MPIVector{U}(U[])
Base.empty!(a::MPIVector) = (empty!(parent(a)); a)
Base.push!(a::MPIVector, x...) = (push!(parent(a), x...); a)    ## TODO: Maybe write a bcast version, when x is only on the root?
Base.append!(a::MPIVector, items) = (append!(parent(a), items); a)
Base.resize!(a::MPIVector, nl::Integer) = (resize!(parent(a), nl); a)
Base.pop!(a::MPIVector) = pop!(parent(a))
Base.sizehint!(a::MPIVector, sz::Integer) = (sizehint!(parent(a), sz); a)

function Base.deleteat!(a::MPIVector, i::Integer)
    deleteat!(a.data, i)
    a
end

function Base.deleteat!(a::MPIVector, inds)
    deleteat!(a.data, inds)
    a
end

function Base.insert!(a::MPIVector, i::Integer, item) ## TODO: Maybe write a bcast version, when x is only on the root?
    insert!(a.data, i, item)
    a
end

end