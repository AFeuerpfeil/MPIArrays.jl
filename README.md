# MPIArrays - Multi-dimensional arrays with fixed size and MPI-safe indexing

<!-- | **Documentation** | **Digital Object Identifier** | **Downloads** |
|:-----------------:|:-----------------------------:|:-------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] |

| **Build Status** | **Coverage** | **Quality assurance** |
|:----------------:|:------------:|:---------------------:|
| [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://github.com/AFeuerpfeil/MPIArrays.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://github.com/AFeuerpfeil/MPIArrays.jl/latest

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.8421339.svg
[doi-url]: https://doi.org/10.5281/zenodo.8421339

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FTensorKit&query=total_requests&label=Downloads
[downloads-url]: http://juliapkgstats.com/pkg/MPIArrays

[ci-img]: https://github.com/AFeuerpfeil/MPIArrays.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/AFeuerpfeil/MPIArrays.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/AFeuerpfeil/MPIArrays.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AFeuerpfeil/MPIArrays.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl -->


MPIArrays.jl is a small package adding the `MPIArray` type which can be backed by any `AbstractArray`. A `MPIArray{T,N,A}` is an `AbstractArray{T,N}` backed by a data array of type `A`. It has a fixed size and features MPI-safe indexing (periodic boundary conditions) across all dimensions: Changing the entries of an `MPIArray`via `setindex!`, `push!`, `append!` etc. automatically broadcasts the entries from the root thus allowing to use MPI parallelization without rewriting code for array access. Due to the overhead of broadcasting each individual change, this package is not suitable for allocating in hot loops. Examples for applications are distributed tensor network algorithms [MPSKitParallel.jl].

The `MPIVector{T}` type is added as an alias for `MPIArray{T, 1}`.

The following constructors are provided.

```julia
# Initialize a MPIArray backed by any AbstractArray.
MPIArray(arr::AbstractArray{T, N}) where {T, N}
# Initialize a MPIArray with default values and the specified dimensions.
MPIArray(initialValue::T, dims...) where T
# Alternative functions for one-dimensional MPI arrays.
MPIVector(arr::AbstractArray{T, 1}) where T
MPIVector(initialValue::T, size::Int) where T
# Alternative functions for two-dimensional MPI arrays.
MPIMatrix(mat::AbstractArray{T, 2}) where T
MPIMatrix(initialValue::T, size::NTuple{2, Integer}) where T
```

### Examples

```julia
julia> using MPIArrays
julia> using MPI
julia> MPI.Init()
julia> a = MPIArray(zeros(Int, 5))
julia> b =  MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? 1 : 2
julia> a[1] = b
 1
 1
```


### License

MPIArrays.jl is licensed under the [MIT license](LICENSE). By using or interacting with this software in any way, you agree to the license of this software.