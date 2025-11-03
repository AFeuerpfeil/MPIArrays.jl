using MPIArrays
using Test
using MPI

# Follow the style used in MPI.jl's tests: initialize MPI at the top-level
# so tests work reliably under mpiexec/mpiexecjl and in single-process runs.
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

@testset "index style" begin
    @test IndexStyle(MPIArray) == IndexCartesian()
    @test IndexStyle(MPIVector) == IndexLinear()
end

@testset "construction" begin
    @testset "construction ($T)" for T = (Float64, Int)
        data = rand(T,10)
        arrays = [MPIVector(data), MPIVector{T}(data),
                MPIArray(data), MPIArray{T}(data), MPIArray{T,1}(data)]
        @test all(a == first(arrays) for a in arrays)
        @test all(a isa MPIVector{T,Vector{T}} for a in arrays)
    end

    @testset "matrix construction" begin
        data = zeros(Float64, 2, 2)
        ref = MPIMatrix(data)
        @test MPIArray{Float64, 2}(data) == ref
        @test MPIMatrix{Float64}(data) == ref
        @test MPIMatrix{Float64, Array{Float64, 2}}(data) == ref
        @test MPIMatrix{Float64, Matrix{Float64}}(data) == ref
    end
end

@testset "getindex / setindex! / push! / append!" begin

    @testset "getindex / setindex! (vector)" begin
        a = MPIVector([0, 0, 0, 0])
        # root supplies authoritative value; other ranks supply different values
        val = rank == 0 ? 10 : 99
        setindex!(a, val, 2)
        # broadcast root's value and compare
        expected = MPI.bcast(rank == 0 ? a[2] : nothing, comm; root=0)
        @test a[2] == expected
    end

    @testset "getindex / setindex! (matrix)" begin
        m = MPIMatrix([1 2; 3 4])
        val = rank == 0 ? 42 : -1
        setindex!(m, val, 1, 2)
        expected = MPI.bcast(rank == 0 ? m[1,2] : nothing, comm; root=0)
        @test m[1,2] == expected
    end

    @testset "getindex / setindex! (non-trivial entries)" begin
        m = MPIMatrix([rand(2,2) for _ in 1:2, _ in 1:3])
        val = rank == 0 ? [9.0 9.0; 9.0 9.0] : [-1.0 -1.0; -1.0 -1.0]
        setindex!(m, val, 1, 2)
        expected = MPI.bcast(rank == 0 ? m[1,2] : nothing, comm; root=0)
        @test m[1,2] == expected
    end

    @testset "push! and append! (MPIVector)" begin
        v = MPIVector(Int[])
        # push!
        item = rank == 0 ? 7 : 999
        push!(v, item)
        ref = MPI.bcast(rank == 0 ? parent(v) : nothing, comm; root=0)
        @test parent(v) == ref

        # append!
        items = rank == 0 ? [10, 11] : [123, 456]
        append!(v, items)
        ref2 = MPI.bcast(rank == 0 ? parent(v) : nothing, comm; root=0)
        @test parent(v) == ref2
    end

    @testset "double MPIArray" begin
        a = MPIArray(MPIArray(rand(Float64, 4)))
        b = rand()
        setindex!(a, b, 1)
        ref = MPI.bcast(rank == 0 ? b : nothing, comm; root=0)
        @test a[1] == ref
    end

    @testset "nested MPIArray" begin
        a = MPIArray([MPIArray(rand(ComplexF64, 4)) for _ in 1:3])
        b = MPIArray(rand(ComplexF64, 4))
        ref = MPI.bcast(rank == 0 ? b : nothing, comm; root=0)
        setindex!(a, b, 1)
        @test a[1] == ref
    end

    @testset "nested MPIArray (2)" begin
        a = MPIArray([MPIArray(rand(ComplexF64, 4)) for _ in 1:3])
        b = rand(ComplexF64)
        ref = MPI.bcast(rank == 0 ? b : nothing, comm; root=0)
        a[1][1] = b
        @test a[1][1] == ref
    end

end


@testset "copy / similar consistency" begin
    base = MPIVector([5,6,7])
    c = copy(base)
    refc = MPI.bcast(rank == 0 ? parent(c) : nothing, comm; root=0)
    @test parent(c) == refc
    s = similar(base, Int, (3,))
    # similar should have same eltype and dims
    @test size(parent(s)) == (3,)
end