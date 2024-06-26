"Efficient array views that consist of multiple windows into another array."
module WindowArrays

# using LoopVectorization: @turbo
using SparseArrays: AbstractSparseArray
using StructArrays: StructArray, StructVector
using FFTW: plan_fft, plan_fft!, plan_rfft, plan_bfft!, plan_brfft,
  FFTW.ESTIMATE, FFTW.MEASURE
using AbstractFFTs: AbstractFFTs.Plan
using ThreadSafeDicts: ThreadSafeDict
using LinearAlgebra: Adjoint, Transpose
import LinearAlgebra
import Base.Threads
# import Statistics

export
  AbstractWindowArray,
  WindowMatrix,
  materialize,
  compact,
  compact!,
  populate_fft_plan_cache,
  mul_prepare,
  BiasArray,
  BiasMatrix

global const n_threads = Ref{Float64}(-1.)

"""
    WindowArrays.num_threads()

Returns the currently configured reference number of threads for array
operations on `AbstractWindowArray`s.

Also see the documentation for the one-argument version.
"""
function num_threads()
  x = n_threads[]
  return (isnan(x) || signbit(x)) ? convert(Float64, Threads.nthreads()) : x
end

"""
    WindowArrays.num_threads(x::Real)

Sets the reference number of threads for array operations on
`AbstractWindowArray`s. If `x` is negative or NaN, `Threads.nthreads()` will
be used. This is also the intial default state.

!!! note
    This number only serves as a reference to determine the actual number of
    asynchronous tasks to be used during an operation. It is a `Float64` and
    can be set to non-integer values. To guarantee single-threaded operation,
    set it to zero.

Returns `WindowArrays.num_threads()`.
"""
function num_threads(x::Real)
  _x = convert(Float64, x)
  n_threads[] = (isnan(_x) || signbit(_x)) ? -1. : _x
  return num_threads()
end

# Helper type to partition reductions into tree structures
struct ReductionTree{I <: Integer}
  in_indexes::UnitRange{I}
  out_index::Int
  task_index::Int
  children::Vector{ReductionTree{I}}

  ReductionTree{I}(in_indexes::AbstractUnitRange, out_index::Int = 0,
      task_index::Int = 1, children = ReductionTree{I}[]
      ) where {I <: Integer} =
    new{I}(convert(UnitRange{I}, in_indexes), out_index, task_index,
      children isa Vector{ReductionTree{I}} ?
        children : ReductionTree{I}[children...])
end

ReductionTree(in_indexes::AbstractUnitRange{I}, args...) where {I <: Integer} =
  ReductionTree{I}(in_indexes, args...)

function ReductionTree(fanout::Real, in_indexes::AbstractUnitRange{<:Integer},
    n_threads::Real = num_threads(), threading_granularity_factor::Real = 0.,
    out_index::Int = 0, next_out_index::Int = 1, task_index::Int = 1,
    n::Int = length(in_indexes))
  I = eltype(in_indexes)
  n_threads *= fanout ^ threading_granularity_factor
  use_threading = n_threads > 1

  if n ≤ fanout
    return ReductionTree(in_indexes, out_index, task_index)
  else
    n_sublevels = floor(log(fanout, n - 1))
    balanced = true
    partition_size = convert(I, ceil(balanced ?
      n / ceil(n / fanout ^ n_sublevels) : fanout ^ n_sublevels))
    n_partitions = ((length(in_indexes) - 1) ÷ partition_size) + 1
    first_partition_end_index =
      last(in_indexes) - partition_size * (n_partitions - 1)
    partition_end_indexes =
      first_partition_end_index : partition_size : last(in_indexes)

    if use_threading
      threads_per_partition = ceil(n_threads / n_partitions)
      threads_for_first_partition = balanced ? threads_per_partition :
        n_threads - threads_per_partition * (n_partitions - 1)
    end

    rt = ReductionTree{I}(true:false, out_index, task_index,
      Vector{ReductionTree{I}}(undef, n_partitions))
    next_next_out_index = next_out_index + (use_threading ? n_partitions : 1)
    for (i, partition_end_index) in enumerate(partition_end_indexes)
      if i > 1
        partition_in_indexes = (partition_end_index - partition_size + one(I) :
          partition_end_index)
        partition_n = convert(Int, partition_size)
        next_n_threads = use_threading ? threads_per_partition : one(n_threads)
      else
        partition_in_indexes = first(in_indexes) : partition_end_index
        partition_n = length(partition_in_indexes)
        next_n_threads =
          use_threading ? threads_for_first_partition : one(n_threads)
      end
      rt.children[i] = ReductionTree(fanout, partition_in_indexes,
        next_n_threads, zero(threading_granularity_factor),
        use_threading ? next_out_index + i - 1 : next_out_index,
        next_next_out_index, task_index, partition_n)
      if use_threading
        last_child = get_last_child(rt.children[i])
        next_next_out_index = last_child.out_index + 1
        task_index = last_child.task_index + 1
      end
    end
    return rt
  end
end

Base.eltype(rt::ReductionTree) = eltype(rt.in_indexes)

get_last_child(rt::ReductionTree) =
  isempty(rt.children) ? rt : get_last_child(last(rt.children))
get_last_out_index(rt::ReductionTree) = get_last_child(rt).out_index
get_last_task_index(rt::ReductionTree) = get_last_child(rt).task_index

function Base.show(io::IO, rt::ReductionTree)
  show(io, typeof(rt))
  print(io, "(")
  show(io, rt.in_indexes)
  print(io, ", ")
  show(io, rt.out_index)
  print(io, ", ")
  show(io, rt.task_index)
  print(io, ", ")
  if get(io, :compact, false)
    show(io, rt.children)
  else
    show(io, eltype(rt.children))
    print(io, "[")
    is_first_entry = true
    for child in rt.children
      if is_first_entry
        is_first_entry = false
      else
        print(io, ",")
      end
      print(io, "\n  ")
      print(io, replace(repr(child; context = io), "\n" => "\n  "))
    end
    print(io, "]")
  end
  print(io, ")")
end

Base.:(==)(rt0::ReductionTree, rt1::ReductionTree) =
  rt0.out_index == rt1.out_index &&
  rt0.task_index == rt1.task_index &&
  rt0.in_indexes == rt1.in_indexes &&
  rt0.children == rt1.children

"""
    reduce(step, combine, rt, init, parameters, out,
      buffer = similar(out, (size(out)..., get_last_out_index(rt))))

Perform reduction according to `ReductionTree` rt.

`size(buffer)` should be equal to `(size(out)..., get_last_out_index(rt))`.

`combine(buffer_slice, new_result)` should be a mutating function that accumulates `new_result` into `buffer_slice`. It could e.g. be defined as
`buffer_slice .+= new_result`. Calling `combine` on the same `buffer_slice`
with multiple `new_result`s should yield (roughly) the same result, independent
of the order in which the calls are made.

`step(buffer_slice, p, task_index)` should be a mutating closure that takes any
value `p` from `parameters`, does any arbitrary computation on it to produce a
`new_result`, and then accumulates that into `buffer_slice` just like `combine`.
I.e., for an arbitrary function `f`, it could be defined as follows:
    new_result = f(p, task_index)
    combine(buffer_slice, new_result)

!!! note
    The reason `step` is used instead of passing `f` directly is that there may
    be efficient ways to perform `f` and `combine` in one go.

A `buffer_slice` in the above will either be `out` or a slice of `buffer` with
the same shape as `out`, as selected by the `out_index` of the `ReductionTree`
node.

Buffer slices (but not `out`) will always be initialized to `init` before any
accumulation steps are performed.

The `task_index` passed to `step` identifies the `Task` that is running the
function call. The following properties hold:
* No two concurrently running tasks will pass the same `task_index` to `step`.
* `task_index` will be an `Integer` between 1 and the total number of tasks
  invoked during the `reduce` call. This number can be obtained by calling
  `get_last_task_index(rt)`.

The end result is the full reduction over all `parameters` and will be saved
into `out`.
"""
function Base.reduce(step::Function, combine::Function, rt::ReductionTree,
    init, parameters::AbstractArray, out::AbstractArray, buffer::AbstractArray =
      similar(out, (size(out)..., get_last_out_index(rt))))
  # Avoid views and construct arrays to the underlying data if it is safe to do
  # so, because it allows for better optimization.
  colons = ntuple(_ -> (:), ndims(out))

  GC.@preserve buffer begin
    make_closure(child) = function()
        slice = unsafe_slice(buffer, (colons..., child.out_index))
        slice .= init
        return reduce(step, combine, child, init, parameters, slice, buffer)
    end

    tasks = [let t = Task(make_closure(child))
        t.sticky = false
        schedule(t)
      end for child in rt.children if child.task_index ≠ rt.task_index]

    for child in rt.children
      if child.task_index == rt.task_index
        combine(out, make_closure(child)())
      end
    end

    for task in tasks
      combine(out, fetch(task))
    end
  end

  @inbounds for in_index in rt.in_indexes
    step(out, parameters[in_index], rt.task_index)
  end

  return out
end

# Helper function to avoid some of the performance deficits of `view`s
@inline function unsafe_slice(a::A, args::Ts
    ) where {N, T, A <: AbstractArray{T, N}, Ts <: Tuple}
  wrap(a, index, size) = unsafe_wrap(Array, pointer(a, index), size)

  if A <: Array && isbitstype(T)
    # TODO: Cases are not at all exhaustive. Ths function could be put somewhere
    # else entirely and made a lot more general. Or maybe `view`s will perform
    # better in the future.
    if Ts <: Tuple{<:AbstractUnitRange}
      r0, = args
      return wrap(a, first(r0), length(r0))
    elseif Ts <: Tuple{<:AbstractUnitRange, Int}
      r0, i1 = args
      return wrap(a, first(r0) + (i1 - 1) * prod(Base.front(size(a))),
        length(r0))
    elseif Ts <: Tuple{Colon, Int}
      c0, i1 = args
      return wrap(a, 1 + (i1 - 1) * prod(Base.front(size(a))),
        Base.front(size(a)))
    elseif Ts <: Tuple{Colon, <:AbstractUnitRange}
      c0, r1 = args
      return wrap(a, 1 + (i1 - 1) * prod(Base.front(size(a))),
        (Base.front(size(a)), length(r1)))
    end
  end
  return @inbounds view(a, args...)
end

abstract type AbstractWindowArray{T, N} <: AbstractArray{T, N} end

"Precomputed data for efficient `LinearAlgebra.mul!(_::AbstractVector,
_::WindowMatrix, _::AbstractVector, _, _)` and
LinearAlgebra.mul!(_::AbstractVector, _::LinearAlgebra.Adjoint{T,
<:WindowMatrix}, _::AbstractVector, _, _) where {T}`"
struct WindowMatrixMulPreparation{T,
  WS <: AbstractMatrix{<:Number},
  ISS <: AbstractArray,
  BP <: Union{Nothing, <:Plan, <:AbstractMatrix{<:Number}},
  BRP <: Union{Nothing, <:Plan, <:AbstractMatrix{<:Number}},
  IP <: Union{Nothing, <:Plan, <:AbstractMatrix{<:Number}},
  IRP <: Union{Nothing, <:Plan, <:AbstractMatrix{<:Number}},
  I <: Integer}
  ws::WS
  iss::ISS
  bp::BP
  brp::BRP
  ip::IP
  irp::IRP
  rt::ReductionTree{I}
  n::Int
  k::Int
end

WindowMatrixMulPreparation{T}(ws, iss, bp, brp, ip, irp, rt, n, k
  ) where {T} = WindowMatrixMulPreparation{T,
      map(typeof, (ws, iss, bp, brp, ip, irp))..., eltype(rt)}(
    ws, iss, bp, brp, ip, irp, rt, n, k)

# TODO: There is an implicit requirement for one-based indexing of `a.v` and
# `a.i` of a `WindowMatrix` `a`. Either make it explicit or re-write the code
# such that the requirement vanishes.

"Matrix whose rows are views into a vector."
struct WindowMatrix{T,
                    V<:AbstractArray{T},
                    I<:AbstractArray{Int},
                    MP<:Union{Nothing, <:WindowMatrixMulPreparation},
                    MPA<:Union{Nothing, <:WindowMatrixMulPreparation}
                   } <: AbstractWindowArray{T, 2}
  v::V
  i::I
  k::Int
  mp::MP
  mpa::MPA
end

WindowMatrix(v, i, k) = WindowMatrix(v, i, k, nothing, nothing)

function Base.size(a::WindowMatrix)
  return (length(a.i), a.k)
end

function Base.eltype(::Type{WindowMatrix{T}}) where {T}
  return T
end

@inline function Base.getindex(a::WindowMatrix, i0::Int, i1::Int)
  return a.v[a.i[i0] + i1 - 1]
end

@inline function Base.getindex(a::WindowMatrix, ::Colon, i1::Int)
  return a.v[vec(a.i) .+ (i1 - 1)]
end

@inline function Base.getindex(a::WindowMatrix, i0::Int, ::Colon)
  return a.v[(0:(a.k-1)) .+ a.i[i0]]
end

@inline function Base.getindex(a::WindowMatrix, is::AbstractVector{Int},
    ::Colon)
  return WindowMatrix(a.v, a.i[is], a.k)
end

@inline function Base.getindex(a::WindowMatrix, ::Colon, r::UnitRange)
  return WindowMatrix(a.v, a.i .+ (r.start - 1), length(r))
end

function Base.view(a::WindowMatrix, i0::Int, ::Colon)
  return view(a.v, (0:(a.k-1)) .+ a.i[i0])
end

function Base.view(a::WindowMatrix, i0::Int, r::UnitRange)
  return view(a.v, r .- 1 .+ a.i[i0])
end

function Base.copyto!(a0::AbstractMatrix{T}, a1::WindowMatrix{T}) where {T}
  @inbounds for (i0, i1) in zip(axes(a0, 1), axes(a1, 1))
    view(a0, i0, :) .= view(a1, i1, :)
  end
  return a0
end

function Base.vcat(a::WindowMatrix)
  return a
end

function Base.vcat(
    as::WindowMatrix{T, <:AbstractVector, <:AbstractVector}...) where {T}
  any([size(a, 2) != size(as[1], 2) for a in as]) &&
    throw(ArgumentError("number of columns of each array must match " *
                        "(got $(map(a -> size(a, 2), as)))"))
  v = vcat([a.v for a in as]...)
  ns = vcat([0], cumsum([length(a.v) for a in as])[1:end-1])
  i = vcat([a.i .+ n for (a, n) in zip(as, ns)]...)
  k = as[1].k
  return WindowMatrix(v, i, k)
end

function Base.vcat(as::WindowMatrix{T}...) where {T}
  # TODO: Perhaps implement the general case here. Or leave the `error()`?
  # TODO: Perhaps also implement another special case for `WindowMatrix{T,
  # <:AbstractMatrix, <:AbstractMatrix}`.
  types_str = join(union(string.(typeof.([as...]))), ", ", " and ")
  error("`Base.vcat` not yet implemented for type(s) $(types_str)")
end

function Base.reduce(::typeof(Base.vcat),
    as::AbstractVector{<:WindowMatrix{<:T}}) where {T}
  return reduce(vcat, as; init = WindowMatrix(T[], Int[], size(as[1], 2)))
end

function Base.broadcast(f, a::WindowMatrix)
  return WindowMatrix(broadcast(f, a.v), a.i, a.k)
end

function Base.broadcast(::typeof(identity), a::WindowMatrix)
  return a
end

function Base.broadcast(f, a::WindowMatrix, args...)
  return WindowMatrix(broadcast(f, a.v, args...), a.i, a.k)
end

function Base.broadcast!(f, dest::WindowMatrix, a::WindowMatrix)
  broadcast!(f, dest.v, a.v)
  return dest
end

function Base.broadcast!(f, dest::WindowMatrix, a::WindowMatrix,
    args...)
  broadcast!(f, dest.v, a.v, args...)
  return dest
end

Base.real(a::WindowMatrix) = WindowMatrix(real(a.v), a.i, a.k)
Base.imag(a::WindowMatrix) = WindowMatrix(imag(a.v), a.i, a.k)

Base.real(a::Adjoint{T, <:WindowMatrix}) where {T} =
  WindowMatrix(real(a.parent.v), a.parent.i, a.parent.k)'
Base.imag(a::Adjoint{T, <:WindowMatrix}) where {T} =
  WindowMatrix(-imag(a.parent.v), a.parent.i, a.parent.k)'

Base.real(a::Transpose{T, <:WindowMatrix}) where {T} =
  transpose(WindowMatrix(real(a.parent.v), a.parent.i, a.parent.k))
Base.imag(a::Transpose{T, <:WindowMatrix}) where {T} =
  transpose(WindowMatrix(imag(a.parent.v), a.parent.i, a.parent.k))

# # Caution: This may break other code relying on `StructArray{<:Complex}`,
# # because it does not create a deep copy of `a.re`/`a.im`.
# Base.real(a::StructArray{<:Complex}) = a.re
# Base.imag(a::StructArray{<:Complex}) = a.im

# Helper functions and cache for FFT plan handling:
#
# These serve the purpose of simplifying FFT plan precomputation when multiple
# threads are performing FFT-based `WindowMatrix` multiplications. FFTW's
# planning methods create locks, so computing the plans prior to multi-threaded
# work is sensible.
#
# `colwise_fft` and `colwise_fft!` apply FFT plans to each column of a matrix.
# This is helpful because it allows to re-use precomputed plans on matrices of
# different widths. Brief testing on x86_64 seems to indicate that performance
# is similar to an FFT plan on the whole matrix. This will certainly not
# generalize to all platforms though, otherwise FFTW would likely already do it
# this way. Also, in the case of `colwise_fft`, extra allocations and copying
# are required, unfortunately.

function colwise_fft!(p::Plan, a::AbstractMatrix{<:Number})
  for i in axes(a, 2)
    p * view(a, :, i)
  end
  return a
end

function colwise_fft(p::Plan, a::AbstractMatrix{<:Number})
  return hcat((p * view(a, :, i) for i in axes(a, 2))...)
end

fft_plan_cache =
  ThreadSafeDict{Tuple{Symbol, DataType, Vector{Int}, Vector{Int}, Int}, Plan}()

# Planning functions which need the `d` argument
const dps = [:plan_irfft, :plan_brfft]
const no_d = -1

# Output prefix and verbosity flags for `get_fft_plan`
const gfpv_pf = "`WindowArrays.get_fft_plan`: "

# TODO: Make this an `Enum`?
const gfpv_cached_plan = 0b1
const gfpv_new_plan = 0b10
const gfpv_method_error = 0b100
const gfpv_populate = 0b1000

function get_fft_plan(name::Symbol, a::AbstractArray{dt}, dims = [1];
    d::Integer = no_d, flags = ESTIMATE, kwargs = (),
    populate = false, force = false, method_check = true,
    verbosity_flags = 0b0) where {dt}
  size = [Base.size(a)...]
  dims = Int.([dims...])
  if name ∈ dps
    d == no_d && (d = (size[1] - 1) * 2) # Assume this if `d` is not given.
    d_arg = (d,)
  else
    d = no_d
    d_arg = ()
  end
  k = (name, dt, size, dims, d)
  if !force && haskey(fft_plan_cache, k)
    verbosity_flags & gfpv_cached_plan ≠ 0 &&
      println(gfpv_pf * "Using cached plan for $(k).")
    return fft_plan_cache[k]
  else
    verbosity_flags & gfpv_new_plan ≠ 0 &&
      println(gfpv_pf * "Finding new plan for $(k).")
    f = eval(name)
    p = try
      f(a, d_arg..., dims; flags = flags, kwargs...)
    catch e
      e isa MethodError && !method_check && (
        verbosity_flags & gfpv_method_error ≠ 0 &&
          println(gfpv_pf * "Ignoring `MethodError` for $(k).");
        return nothing)
      throw(e)
    end
    if populate
    verbosity_flags & gfpv_populate ≠ 0 &&
      println(gfpv_pf * "Populating plan cache for $(k).")
      force || !haskey(fft_plan_cache, k) && (fft_plan_cache[k] = p)
    end
    return p
  end
end

"""
    populate_fft_plan_cache(name, dt, size, dims = [1]; \
      flags = FFTW.MEASURE, kw...)

Precomputes and caches an FFT plan for input arrays of the given `size` and
element type `dt` along dimensions `dims`. The `Symbol` `name` selects the
planning function to use (e.g. `:plan_brfft` for a backwards real FFT, see
documentation of `AbstractFFTs` and `FFTW` packages).
"""
function populate_fft_plan_cache(name::Symbol, dt::DataType, size, dims = [1];
    flags = MEASURE, kwargs...)
  # For backward real FFTs, interpret `size` as the output size.
  name ∈ dps && (size = [i == 1 ? s ÷ 2 + 1 : s for (i, s) in enumerate(size)])
  # TODO: Infer complex/real from `name` instead of relying on `method_check`.
  a = Array{dt}(undef, size...) # TODO: Is `undef` okay for planning?
  return get_fft_plan(name, a, dims; populate = true, flags = flags, kwargs...)
end


"""
    populate_fft_plan_cache(names, dts, sizes, dimss = [[1]]; kw...)

All arguments must be iterable. Forms the cartesian product of the arguments and
precomputes the FFT plans for all combinations.
"""
function populate_fft_plan_cache(names, dts, sizes, dimss = [[1]];
    method_check = false, kwargs...)
  return [populate_fft_plan_cache(name, dt, size, dims;
                                  method_check = method_check, kwargs...)
    for (name, dt, size, dims) in Iterators.product(names, dts, sizes, dimss)]
end

for name in [:plan_fft, :plan_fft!, :plan_rfft, :plan_bfft!, :plan_brfft]
  fname = Symbol("_$(name)")
  sname = "$(name)"
  if name ∈ dps
    def = :(function $(fname)(a, d, dims = [1]; kwargs...)
              get_fft_plan(Symbol($(sname)), a, dims; d = d, kwargs...)
            end)
  else
    def = :(function $(fname)(a, dims = [1]; kwargs...)
              get_fft_plan(Symbol($(sname)), a, dims; kwargs...)
            end)
  end
  eval(def)
end

# TODO: `adjvec = Val(false)` currently, because the FFT-based adjoint-vector
# multiplication produces faulty results. If it gets fixed at some point in the
# future, it should be enabled by default again. And the note in the docstring
# should be removed.
# TODO: This function should probably be broken down into smaller parts
# TODO: Type stability
"""
    mul_prepare(a, matvec = Val(true), adjvec = Val(false), \
      return_plans = Val(false); kwargs...)

Returns a `WindowMatrix` identical to `a`, but with the fields `mp` and `mpa`
updated to prepare for FFT-based matrix-vector and adjoint-vector
multiplication.

!!! note
    `adjvec = Val(true)` produces faulty results currently.
    There is an open to-do item about this in `$(@__FILE__)`.
    It is okay to just leave it disabled for now, because there exists a
    non-FFT-based multiplication method that is relatively efficient as well.

# Arguments

* `a`: Input `WindowMatrix`.
* `matvec::Val`: Whether to update `a.mp` for matrix-vector multiplication.
* `adjvec::Val`: Whether to update `a.mpa` for adjoint-vector multiplication.
* `return_plans::Val`: Whether to return the FFT plans for use in a call to
  `mul_prepare` on a similar `WindowMatrix`. This way, FFT planning only needs
  to be done once. Keep in mind that `FFTW` planning routines will lock threads,
  even for `FFTW.ESTIMATE`, it seems.

# Keyword arguments:

Some of these arguments set the value for multiple related arguments, which can
also be set directly to provide more fine-grained control.

* `*_minimal_payload_size`: Smallest allowed overlap-save payload size.
* `flags`: Sets the default flags for all FFT planning routines. Set `*_flags`
  for more fine-grained control.
* `type`: Either `Real` or `Complex`. Set to `Real` to do an `AbstractFFTs.rfft`
  on the contents of `a.v`. This improves efficiency when `a` and the
  multiplication vectors are real-valued. Default depends on `eltype(a)`. Set
  `*_type` for more fine-grained control.
* `plan_*p::Bool`: Whether to create FFT plans for specific cases. Default
  depends on `eltype(a)`. Set `*_plan_*p` for more fine grained control.
* `*_*p`: Provide specific FFT plans to avoid re-planning.
* `ai_is_sorted::Bool`: Whether `a.i` is sorted. Default: `issorted(a.i)`.
"""
function mul_prepare(a::WindowMatrix{T},
    ::Val{matvec} = Val(true), ::Val{adjvec} = Val(false),
    ::Val{return_plans} = Val(false);
    # TODO: Improve heuristics for the payload sizes?
    matvec_minimal_payload_size = 9a.k, adjvec_minimal_payload_size = 3a.k,
    fanout = 16, n_threads = num_threads(), threading_granularity_factor = 0.,
    matvec_fanout = fanout, adjvec_fanout = fanout,
    matvec_n_threads = n_threads, adjvec_n_threads = n_threads,
    matvec_threading_granularity_factor = threading_granularity_factor,
    adjvec_threading_granularity_factor = threading_granularity_factor,
    flags = ESTIMATE, mul_flags = flags,
    bp_flags = mul_flags, brp_flags = mul_flags, ip_flags = mul_flags,
    irp_flags = mul_flags, wp_flags = flags,
    matvec_bp_flags = bp_flags, adjvec_bp_flags = bp_flags,
    matvec_brp_flags = brp_flags, adjvec_brp_flags = brp_flags,
    matvec_ip_flags = ip_flags, adjvec_ip_flags = ip_flags,
    matvec_irp_flags = irp_flags, adjvec_irp_flags = irp_flags,
    matvec_wp_flags = wp_flags, adjvec_wp_flags = wp_flags,
    type = T <: Real ? Real : Complex, matvec_type = type, adjvec_type = type,
    plan_bp = T <: Complex, plan_brp = T <: Real,
    plan_ip = T <: Complex, plan_irp = T <: Real,
    matvec_plan_bp = plan_bp, matvec_plan_brp = plan_brp,
    matvec_plan_ip = plan_ip, matvec_plan_irp = plan_irp,
    adjvec_plan_bp = plan_bp, adjvec_plan_brp = plan_brp,
    adjvec_plan_ip = plan_ip, adjvec_plan_irp = plan_irp,
    matvec_bp = nothing, matvec_brp = nothing, matvec_ip = nothing,
    matvec_irp = nothing, matvec_wp = nothing,
    adjvec_bp = nothing, adjvec_brp = nothing, adjvec_ip = nothing,
    adjvec_irp = nothing, adjvec_wp = nothing,
    ai_is_sorted = issorted(a.i), gfpv_flags = 0b0
    ) where {T <: Number, matvec, adjvec, return_plans}
  is_complex = T <: Complex
  CT = complex(T)

  @assert !(matvec_type == Real && is_complex) "`rfft` requested for complexes."
  @assert !(adjvec_type == Real && is_complex) "`rfft` requested for complexes."

  if !ai_is_sorted
    ai_sortperm = sortperm(a.i)
    ais = a.i[ai_sortperm]
  else
    ais = a.i
  end

  if matvec
    if length(a) == 0
      k = 0
      n = a.k
      ws = Matrix{T}(undef, 0, 0)
      iss = []
      buf = Vector{CT}(undef, n)
      rbuf = Vector{real(CT)}(undef, n)
      kp = Matrix{T}(undef, 0, a.k)
      isnothing(matvec_bp)  && matvec_plan_bp  && (matvec_bp  = kp)
      isnothing(matvec_brp) && matvec_plan_brp && (matvec_brp = kp)
      # Keep `ip` and `irp` at `nothing` unless given: nothing to do.
    else
      n = 2^Int(ceil(log2(matvec_minimal_payload_size + a.k - 1))) # Window size
      p = n - a.k + 1 # Payload size
      m = Int(ceil(length(a.v) // p)) # Number of windows
      k = n ÷ 2 + 1 # Length of an `rfft` result on a buffer of length `n`

      # TODO: There's probably a faster way of doing this, using `ais`.
      iss = let
        iss = [StructVector(i = Int[], buf_i = Int[]) for _ in 1:m]
        for i in axes(a.i, 1)
          ai = a.i[i]
          j_min = Int( ceil((ai - p) // p)) + 1
          j_max = Int(floor((ai + n - p - 1) // p)) + 1
          for j in intersect(j_min:j_max, 1:m)
            # TODO: This could be done without the circular wrapping – it's just
            # a question of the payload's placement in the windows, right?
            buf_i0 = (j - 1)p + 1
            buf_i = (ai < buf_i0 ? 0 : n) + buf_i0 - ai
            push!(iss[j], (i = i, buf_i = buf_i))
          end
        end
        iss
      end

      ws = let # Compute reversed windows into `a.v` to be FFT'd
        # Note: This could also be done with `WindowMatrix` and `PaddedView`,
        # but let's do it low-level
        ws_t = matvec_type == Real ? real(CT) : CT
        ws = Matrix{ws_t}(undef, n, m)
        view(ws, 1 : n - p, :) .= zero(ws_t)
        is = range(1; step = p, length = m)
        for i in 1 : m - 1
          r = range(is[i]; length = p)
          view(ws, n - p + 1 : n, i) .= view(a.v, r) .* inv(n)
          reverse!(view(ws, n - p + 1 : n, i))
        end
        rest = view(a.v, is[m] : length(a.v)) ./ n
        view(ws, n : -1 : n - length(rest) + 1, m) .= rest
        view(ws, n - p + 1 : n - length(rest), m) .= zero(ws_t)
        ws
      end

      isnothing(matvec_wp) && (matvec_wp = (matvec_type == Real ?
        _plan_rfft : _plan_fft!)(view(ws, :, 1);
        flags = matvec_wp_flags, verbosity_flags = gfpv_flags))
      ws = (matvec_type == Real ? colwise_fft : colwise_fft!)(matvec_wp, ws)

      buf = Vector{CT}(undef, n)
      rbuf = Vector{real(CT)}(undef, n)
      irbuf = view(buf, 1 : k)

      isnothing(matvec_bp)  && matvec_plan_bp  && (matvec_bp  = _plan_fft(
        buf; flags = matvec_bp_flags, verbosity_flags = gfpv_flags))
      isnothing(matvec_brp) && matvec_plan_brp && (matvec_brp = _plan_rfft(
        rbuf; flags = matvec_brp_flags, verbosity_flags = gfpv_flags))
      isnothing(matvec_ip)  && matvec_plan_ip  && (matvec_ip  = _plan_bfft!(
        buf; flags = matvec_ip_flags, verbosity_flags = gfpv_flags))
      isnothing(matvec_irp) && matvec_plan_irp && (matvec_irp = _plan_brfft(
        irbuf, n; flags = matvec_irp_flags, verbosity_flags = gfpv_flags))
    end
    rt = ReductionTree(matvec_fanout, axes(ws, 2), matvec_n_threads,
      matvec_threading_granularity_factor)
    mp = WindowMatrixMulPreparation{matvec_type}(ws, iss,
      matvec_bp, matvec_brp, matvec_ip, matvec_irp, rt, n, k)
    matvec_plans = ((matvec_bp, matvec_brp, matvec_ip, matvec_irp, matvec_wp),)
  else
    mp = a.mp
    matvec_plans = ()
  end

  if adjvec
    if length(a) == 0
      k = 0
      n = 0
      ws = Matrix{T}(undef, 0, 0)
      iss = []
      buf = Vector{CT}(undef, 0)
      rbuf = Vector{real(CT)}(undef, 0)
      # Keep FFT plans at `nothing` unless given: nothing to do.
    else
      n = 2^Int(ceil(log2(max(adjvec_minimal_payload_size, a.k)))) # Window size
      k = n ÷ 2 + 1 # Length of an `rfft` result on a buffer of length `n`

      iss = let
        iss = NamedTuple{(:b_i, :buf_i), Tuple{UnitRange{Int}, Vector{Int}}}[]
        i0 = 1
        while !isnothing(i0)
          i1 = findnext(ai -> ai + a.k - 1 > ais[i0] + n - 1, ais, i0 + 1)
          b_i = i0 : (isnothing(i1) ? length(ais) : i1 - 1)
          buf_i = (i -> i == 0 ? 1 : (n - i + 1)).(view(ais, b_i) .- ais[i0])
          push!(iss, (b_i = b_i, buf_i = buf_i))
          i0 = i1
        end
        iss
      end

      m = length(iss) # Number of windows

      ws = let # Compute windows into `a.v` to be FFT'd
        ws_t = adjvec_type == Real ? real(CT) : CT
        ws = Matrix{ws_t}(undef, n, m)
        for i in 1:m
          i0, i1 = ais[iss[i].b_i.start], ais[iss[i].b_i.stop] + a.k - 1
          view(ws, 1 : i1 - i0 + 1, i) .= conj.(a.v[i0:i1]) ./ n
          view(ws, i1 - i0 + 2 : size(ws, 1), i) .= zero(ws_t)
        end
        ws
      end

      if !ai_is_sorted
        issp = iss
        iss = NamedTuple{(:b_i, :buf_i), Tuple{Vector{Int}, Vector{Int}}}[]
        for (b_i, buf_i) in issp
          push!(iss, (b_i = ai_sortperm[b_i], buf_i = buf_i))
        end
      end

      isnothing(adjvec_wp) && (adjvec_wp = (adjvec_type == Real ?
        _plan_rfft : _plan_fft!)(view(ws, :, 1);
        flags = adjvec_wp_flags, verbosity_flags = gfpv_flags))
      ws = (adjvec_type == Real ? colwise_fft : colwise_fft!)(adjvec_wp, ws)

      buf = Vector{CT}(undef, n)
      rbuf = Vector{real(CT)}(undef, n)
      irbuf = view(buf, 1 : k)

      isnothing(adjvec_bp)  && adjvec_plan_bp  && (adjvec_bp  = _plan_fft!(
        buf; flags = adjvec_bp_flags, verbosity_flags = gfpv_flags))
      isnothing(adjvec_brp) && adjvec_plan_brp && (adjvec_brp = _plan_rfft(
        rbuf; flags = adjvec_brp_flags, verbosity_flags = gfpv_flags))
      isnothing(adjvec_ip)  && adjvec_plan_ip  && (adjvec_ip  = _plan_bfft!(
        buf; flags = adjvec_ip_flags, verbosity_flags = gfpv_flags))
      isnothing(adjvec_irp) && adjvec_plan_irp && (adjvec_irp = _plan_brfft(
        irbuf, n; flags = adjvec_irp_flags, verbosity_flags = gfpv_flags))
    end

    rt = ReductionTree(adjvec_fanout, axes(ws, 2), adjvec_n_threads,
      adjvec_threading_granularity_factor)
    mpa = WindowMatrixMulPreparation{adjvec_type}(ws, iss,
      adjvec_bp, adjvec_brp, adjvec_ip, adjvec_irp, rt, n, k)
    adjvec_plans = ((adjvec_bp, adjvec_brp, adjvec_ip, adjvec_irp, adjvec_wp),)
  else
    mpa = a.mpa
    adjvec_plans = ()
  end

  a_mp = WindowMatrix(a.v, a.i, a.k, mp, mpa)
  return return_plans ? (a_mp, matvec_plans..., adjvec_plans...) : a_mp
end

# `mul!` computes c <- a b α + c β

@inline function LinearAlgebra.mul!(
    c::AbstractVector{<:Number},
    a::WindowMatrix{<:Number, <:Any, <:Any, <:WindowMatrixMulPreparation},
    b::AbstractVector{<:Number},
    α::Number, β::Number)
  ((size(c, 1), size(b, 1)) == size(a) && size(c, 2) == size(b, 2)) ||
    throw(DimensionMismatch("Cannot multiply matrix of size $(size(a)) with \
    matrix of size $(size(b)) to yield a result size of $(size(c))"))

  inputs_are_real = eltype(a) <: Real && eltype(b) <: Real
  mp_is_real = a.mp isa WindowMatrixMulPreparation{Real}
  n_tasks = get_last_task_index(a.mp.rt)

  buf = Array{complex(eltype(a))}(undef, a.mp.n, n_tasks)

  c .*= β

  dft_b_buf = inputs_are_real ? Array{real(eltype(a))}(undef, a.mp.n) :
    view(buf, :, 1)
  @inbounds view(dft_b_buf, Base.OneTo(a.k)) .= b
  @inbounds view(dft_b_buf, a.k + 1 : length(dft_b_buf)) .=
    zero(eltype(dft_b_buf))
  dft_b = (inputs_are_real ? a.mp.brp : a.mp.bp) * dft_b_buf

  GC.@preserve buf a begin
    @inline step(c, i, ti) = @inbounds begin
      slice = unsafe_slice(buf, (:, ti))
      if inputs_are_real
        unsafe_slice(slice, (Base.OneTo(a.mp.k),)) .=
          dft_b .* unsafe_slice(a.mp.ws, (Base.OneTo(a.mp.k), i))
        partial_result = a.mp.irp * unsafe_slice(slice, (Base.OneTo(a.mp.k),))
      else
        if !mp_is_real
          slice .= dft_b .* unsafe_slice(a.mp.ws, (:, i))
        else
          unsafe_slice(slice, (Base.OneTo(a.mp.k),)) .= (*).(
            unsafe_slice(dft_b, (Base.OneTo(a.mp.k),)),
            unsafe_slice(a.mp.ws, (:, i)))
          unsafe_slice(slice, (a.mp.k + 1 : a.mp.n,)) .= (*).(
            unsafe_slice(dft_b, (a.mp.k + 1 : length(dft_b),)),
            conj.(unsafe_slice(a.mp.ws, (a.mp.k - 1 : -1 : 2, i))))
        end
        partial_result = a.mp.ip * slice
      end
      _copy_els(c, a.mp.iss[i].i, partial_result, a.mp.iss[i].buf_i,
        (x, y) -> muladd(α, y, x))
    end

    @inline combine(out, in) = out .+= in

    c = reduce(step, combine, a.mp.rt, zero(eltype(c)), axes(a.mp.ws, 2), c)
  end
  return c
end

# TODO: At the moment, this FFT-based adjoint-vector multiplication method
# produces faulty results. It might need to be rewritten from scratch, together
# with the part of `mul_prepare` that precomputes the data needed for this
# method.
# As a conceptual starting point: The multiplication has to convolve
# `a.parent.v` with a sparse vector whose elements are determined from `b`,
# and their indexes are determined by `a.i`. Only a small portion of the whole
# convolved signal has to be computed, which is why it should be possible to
# implement the otherwise large convolution efficiently.
# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{<:Number},
#     a::Ad,
#     b::AbstractVector{B},
#     α::Number, β::Number) where {A<:Number, B<:Number,
#       MPA<:WindowMatrixMulPreparation,
#       Ad<:Union{  <:Adjoint{A, <:WindowMatrix{A, <:Any, <:Any, <:Any, MPA}},
#                 <:Transpose{A, <:WindowMatrix{A, <:Any, <:Any, <:Any, MPA}}}}
#   # Profiling results with FFTW.MEASURE compared to BLAS on one thread, using
#   # feature matrix for S5, BPSK45, and running `a * b`:
#   # On lucille (ppc64le) for A, B = Float32:    ~1.2x faster, but bad numerics
#   # On lucille (ppc64le) for A, B = Float64:    ~1.3x slower, and allocates
#   # On lucille (ppc64le) for A, B = ComplexF32: ~1.9x faster, so-so numerics
#   # On lucille (ppc64le) for A, B = ComplexF64: ~1.1x faster

#   inputs_are_real = A <: Real && B <: Real
#   mpa_is_real = MPA <: WindowMatrixMulPreparation{Real}
#   is_transpose = Ad <: Transpose

#   a = a.parent

#   c .*= β

#   _buf = inputs_are_real ? a.mpa.rbuf : a.mpa.buf
#   _bp  = inputs_are_real ? a.mpa.brp  : a.mpa.bp
#   _ip  = inputs_are_real ? a.mpa.irp  : a.mpa.ip

#   @inbounds for i in axes(a.mpa.ws, 2)
#     # Note: Dead branches should be optimized away by the compiler here.
#     _buf .= zero(eltype(_buf))
#     _copy_els(_buf, a.mpa.iss[i].buf_i, b, a.mpa.iss[i].b_i,
#       is_transpose ? (_, x) -> conj(x) : (_, x) -> x)

#     buf = _bp * _buf
#     if inputs_are_real
#       buf .*= view(a.mpa.ws, Base.OneTo(a.mpa.k), i)
#     else
#       if !mpa_is_real
#         buf .*= view(a.mpa.ws, :, i)
#       else
#         view(buf, Base.OneTo(a.mpa.k)) .*= view(a.mpa.ws, :, i)
#         view(buf, a.mpa.k + 1 : length(buf)) .*=
#           conj.(view(a.mpa.ws, a.mpa.k - 1 : -1 : 2, i))
#       end
#     end
#     buf = _ip * buf

#     if is_transpose
#       c .+= conj.(view(buf, Base.OneTo(a.k))) .* α
#     else
#       c .+= view(buf, Base.OneTo(a.k)) .* α
#     end
#   end
#   return c
# end

@inline function LinearAlgebra.mul!(
    c::AbstractVector{<:Number}, a::Ad, b::AbstractVector{<:Number},
    α::Number, β::Number;
    fanout::Real = 8, n_threads::Real = num_threads(),
    threading_granularity_factor::Real = -.25) where {
      WM <: WindowMatrix{<:Number, <:Any, <:Any, <:Any, Nothing},
      Ad <: Union{<:Adjoint{<:Number, WM}, <:Transpose{<:Number, WM}}}
  ((size(c, 1), size(b, 1)) == size(a) && size(c, 2) == size(b, 2)) ||
    throw(DimensionMismatch("Cannot multiply matrix of size $(size(a)) with \
    matrix of size $(size(b)) to yield a result size of $(size(c))"))

  is_transpose = Ad <: Transpose

  a = a.parent
  f = is_transpose ? identity : conj

  c .*= β

  parameters = axes(a, 1)
  rt = ReductionTree(fanout, eachindex(parameters), n_threads,
    threading_granularity_factor)
  @inline step(c, i, _) = let _i = a.i[i]
    c .+= f.(view(a.v, _i : _i + (a.k - 1))) .* (b[i] * α)
  end
  @inline combine(c, buf) = c .+= buf

  return reduce(step, combine, rt, zero(eltype(c)), parameters, c)
end

# Helper function to copy one set of array elements onto a set of another
# array's elements, optionally with additional operations. This is a common
# pattern in some functions here. This helper function eliminates some case
# distinctions between ranges versus vectors of indexes and views versus
# indexing, and somewhat unexpectedly seems to be more efficient in any case
# where at least one of `is` and `js` is some vector of indexes, i.e. not a
# range. Seems to be because indexing usually allocates new arrays.
@inline function _copy_els(xs, is, ys, js, op = (_, y) -> y)
  @inbounds for (i, j) in zip(is, js)
    xs[i] = op(xs[i], ys[j])
  end
end

@inline _copy_els(xs, _::Colon, ys, js, args...) =
  _copy_els(xs, eachindex(xs), ys, js, args...)
@inline _copy_els(xs, is, ys, _::Colon, args...) =
  _copy_els(xs, is, ys, eachindex(ys), args...)

# TODO: `@turbo` methods exhibit significant numerical errors for big matrices,
# probably due to long running sums. Can this be improved by splitting the
# computation into several parts? Need to re-benchmark all methods that depend
# on the `@turbo` implementations after improving their numerics.

# `mul!` computes c <- a b α + c β

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T}, a::WindowMatrix{<:U}, b::AbstractVector{<:U},
#     α::Number, β::Number) where {T<:Real, U<:T}
#   # ~1.2x faster than materialized matrix multiplication on my system
#   acc = zeros(U, size(c))
#   @turbo for i in axes(a, 1), j in axes(a, 2)
#     acc[i] += a.v[a.i[i] + j - 1] * b[j]
#   end
#   return copyto!(c, acc * α + c * β)
# end

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T}, a::WindowMatrix{<:T, <:StructArray, <:Any, Nothing},
#     b::AbstractVector{<:T}, α::Number, β::Number) where {T<:Complex}
#   # # ~9x slower than materialized matrix multiplication on my system
#   # acc_re = zeros(T, size(c))
#   # acc_im = zeros(T, size(c))
#   # b_re, b_im = reim(b)
#   # @turbo for i in axes(a, 1), j in axes(a, 2)
#   #   i0 = a.i[i] + j - 1
#   #   avre, avim = a.v.re[i0], a.v.im[i0]
#   #   bre, bim = b_re[j], b_im[j]
#   #   r = avre * bre - avim * bim
#   #   acc_re[i] += r
#   #   acc_im[i] += (avre + avim) * (bre + bim) - r
#   # end
#   # return copyto!(c, (acc_re + acc_im * im) * α + c * β)

#   # ~2.3x slower than materialized matrix multiplication on my system
#   a_re = WindowMatrix(a.v.re, a.i, a.k)
#   a_im = WindowMatrix(a.v.im, a.i, a.k)
#   b_re, b_im = reim(b)
#   c_re, c_im = reim(c)

#   c_re = LinearAlgebra.mul!(c_re, a_re, b_re, α, false)
#   c_re = LinearAlgebra.mul!(c_re, a_im, b_im, α, true)
#   c_im = LinearAlgebra.mul!(c_im, a_re, b_im, α, false)
#   c_im = LinearAlgebra.mul!(c_im, a_im, b_re, -α, true)
#   return copyto!(c, c_re + c_im * im + c * β)
# end

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T}, a::WindowMatrix{<:T, <:Any, <:Any, Nothing},
#     b::AbstractVector{<:T},
#     α::Number, β::Number) where {T<:Complex}
#   # ~4.5x slower than materialized matrix multiplication on my system
#   a_re, a_im = reim(a)
#   b_re, b_im = reim(b)
#   c_re, c_im = reim(c)

#   c_re = LinearAlgebra.mul!(c_re, a_re, b_re, α, false)
#   c_re = LinearAlgebra.mul!(c_re, a_im, b_im, -α, true)
#   c_im = LinearAlgebra.mul!(c_im, a_re, b_im, α, false)
#   c_im = LinearAlgebra.mul!(c_im, a_im, b_re, α, true)
#   return copyto!(c, c_re + c_im * im + c * β)
# end

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T}, a::Adjoint{<:U, <:WindowMatrix},
#     b::AbstractVector{<:U}, α::Number, β::Number) where {T<:Real, U<:T}
#   # ~1.6x slower than materialized matrix multiplication on my system
#   acc = zeros(U, size(c))
#   @turbo for i in axes(a.parent, 1), j in axes(a.parent, 2)
#     acc[j] += a.parent.v[a.parent.i[i] + j - 1] * b[i]
#   end
#   return copyto!(c, acc * α + c * β)
# end

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T},
#     a::Adjoint{<:T, <:WindowMatrix{<:T, <:StructArray}},
#     b::AbstractVector{<:T}, α::Number, β::Number) where {T<:Complex}
#   # # ~4.5x slower than materialized matrix multiplication on my system
#   # acc_re = zeros(T, size(c))
#   # acc_im = zeros(T, size(c))
#   # b_re, b_im = reim(b)
#   # @turbo for i in axes(a.parent, 1), j in axes(a.parent, 2)
#   #   i0 = a.parent.i[i] + j - 1
#   #   avre, avim = a.parent.v.re[i0], a.parent.v.im[i0]
#   #   bre, bim = b_re[i], b_im[i]
#   #   r = avre * bre + avim * bim
#   #   acc_re[j] += r
#   #   acc_im[j] += (avre - avim) * (bre + bim) - r
#   # end
#   # return copyto!(c, (acc_re + acc_im * im) * α + c * β)

#   # ~3x slower than materialized matrix multiplication on my system
#   a_re = WindowMatrix(a.parent.v.re, a.parent.i, a.parent.k)'
#   a_im = WindowMatrix(a.parent.v.im, a.parent.i, a.parent.k)'
#   b_re, b_im = reim(b)
#   c_re, c_im = reim(c)

#   c_re = LinearAlgebra.mul!(c_re, a_re, b_re, α, false)
#   c_re = LinearAlgebra.mul!(c_re, a_im, b_im, α, true)
#   c_im = LinearAlgebra.mul!(c_im, a_re, b_im, α, false)
#   c_im = LinearAlgebra.mul!(c_im, a_im, b_re, -α, true)
#   return copyto!(c, c_re + c_im * im + c * β)
# end

# @inline function LinearAlgebra.mul!(
#     c::AbstractVector{T}, a::Adjoint{<:T, <:WindowMatrix},
#     b::AbstractVector{<:T}, α::Number, β::Number) where {T<:Complex}
#   # ~5x slower than materialized matrix multiplication on my system
#   a_re, a_im = reim(a)
#   b_re, b_im = reim(b)
#   c_re, c_im = reim(c)

#   c_re = LinearAlgebra.mul!(c_re, a_re, b_re, α, false)
#   c_re = LinearAlgebra.mul!(c_re, a_im, b_im, -α, true)
#   c_im = LinearAlgebra.mul!(c_im, a_re, b_im, α, false)
#   c_im = LinearAlgebra.mul!(c_im, a_im, b_re, α, true)
#   return copyto!(c, c_re + c_im * im + c * β)
# end

# Custom methods for `Base.sum` with `WindowMatrix`. These rely on the
# `LinearAlgebra.mul!` methods for performance and numerics.
# TODO: Adapt code from `mul!` to the case of an all-ones vector `b` to further
# increase performance?
function Base.sum(f,
    a::Union{<:WindowMatrix{T},
             <:Adjoint{T, <:WindowMatrix{T}},
             <:Transpose{T, <:WindowMatrix{T}}};
    dims = :) where {T<:Number}
  isempty(a) && return sum(f, materialize(a); dims = dims)

  ds = _canonicalize_dims(AbstractMatrix, dims)
  s = _sum(f, a, ds, _sum_buf(f, a, ds), Val(a isa Adjoint))
  dims isa Colon && return first(s)
  return s
end

# TODO: Is this behavior already given through `Base`?
function Base.sum(a::Union{<:WindowMatrix{T},
                           <:Adjoint{T, <:WindowMatrix{T}},
                           <:Transpose{T, <:WindowMatrix{T}}};
    kwargs...) where {T<:Number}
  return sum(identity, a; kwargs...)
end

function _sum_buf(f, a::AbstractMatrix, dims)
  t = typeof(f(a[1])) # TODO: Improve the way this type promotion is done?
  if 1 ∈ dims && 2 ∈ dims
    return Matrix{t}(undef, 1, 1)
  elseif 1 ∈ dims
    return Matrix{t}(undef, 1, size(a, 2))
  elseif 2 ∈ dims
    return Matrix{t}(undef, size(a, 1), 1)
  end
  return Matrix{t}(undef, 0, 0)
end

function _canonicalize_dims(::Type{<:AbstractMatrix}, dims)
  dims isa Colon && return (1, 2)
  dims isa Number && return (dims,)
  dims isa Tuple && return dims
  return (ds...,)
end

function _sum(f,
    a::Union{<:Transpose{T, <:WindowMatrix{T}},
             <:Adjoint{T, <:WindowMatrix{T}}},
    dims, buf, ::Val{is_conj}) where {T <: Number, is_conj}
  1 ∈ dims && 2 ∈ dims && return _sum(f, a.parent, dims, buf, Val(is_conj))
  1 ∈ dims && return _sum(f, a.parent, (2,), buf, Val(is_conj))
  2 ∈ dims && return _sum(f, a.parent, (1,), buf, Val(is_conj))
  return broadcast(f, a)
end

function _sum(f, a::WindowMatrix{T}, dims, buf, ::Val{is_conj}
    ) where {T <: Number, is_conj}
  f_is_id = typeof(f) == typeof(identity)
  f_conj = T <: Real ? f : (f_is_id ? conj : x -> f(conj(x)))

  if 1 ∈ dims && 2 ∈ dims
    _f = !f_is_id && is_conj ? f_conj : f

    # Heuristically choose fastest method. TODO: Improve heuristics.
    if isnothing(a.mp)
      if isnothing(a.mpa)
        s = sum(_f, a; dims = 2)
      else
        s = sum(_f, a; dims = 1)
      end
    else
      if isnothing(a.mpa)
        s = sum(_f, a; dims = 2)
      else
        if size(a, 1) / size(a, 2) > 10
          s = sum(_f, a; dims = 2)
        else
          s = sum(_f, a; dims = 1)
        end
      end
    end

    buf[1] = sum(s)
    f_is_id && is_conj && (buf .= conj.(buf))
    return buf
  elseif 1 ∈ dims
    !f_is_id && (a = broadcast(is_conj ? f_conj : f, a))
    ta = f_is_id && is_conj ? a' : transpose(a)
    LinearAlgebra.mul!(view(buf, :), ta, ones(eltype(a), size(a, 1)))
    return buf
  elseif 2 ∈ dims
    !f_is_id && (a = mul_prepare(broadcast(is_conj ? f_conj : f, a),
                                 Val(true), Val(false)))
    LinearAlgebra.mul!(view(buf, :), a, ones(eltype(a), size(a, 2)))
    f_is_id && is_conj && (buf .= conj.(buf))
    return buf
  elseif dims isa Tuple{}
    return broadcast(is_conj ? f_conj : f, a)
  end
  # Most common cases have been handled by now, this edge case has to
  # materialize for type stability
  return sum(is_conj ? f_conj : f, materialize(a); dims)
end

# Redirect `mapreduce(f, +, a::WindowMatrix)` to `sum`
function Base.mapreduce(f, ::typeof(+),
    a::Union{<:WindowMatrix{T},
             <:Adjoint{T, <:WindowMatrix{T}},
             <:Transpose{T, <:WindowMatrix{T}}};
    dims = :, init = zero(T)) where {T}
  return init + sum(f, a; dims = dims)
end

# Custom methods for some `Statistics` functions with `WindowMatrix`. These rely
# on the custom `Base.sum` methods for better efficiency. Note: The
# implementation of `mean` in `Statistics` also relies on `Base.sum`, but does
# this type promotion thing which we can avoid in the case where `f` is
# `identity` to run a `Base.sum` with less overhead.
# function Statistics.mean(::typeof(identity),
#     a::Union{<:WindowMatrix{T},
#              <:Adjoint{T, <:WindowMatrix{T}},
#              <:Transpose{T, <:WindowMatrix{T}}};
#     dims = :) where {T<:Number}
# 
#   ds = _canonicalize_dims(AbstractMatrix, dims)
#   n = isempty(ds) ? 0 : prod([size(a, d) for d in ds])
#   w = 1 // n
# 
#   s = sum(a; dims = dims)
#   if s isa Number
#     return s * w
#   else
#     try
#       return s .*= w
#     catch InexactError
#       return s .* w
#     end
#   end
# end

# Note: `Statistics` otherwise redirects to `_mean` directly. We could provide a
# method for `_mean` instead of `mean` here, but since it is not part of the
# exposed API, it may be a bad idea.
# function Statistics.mean(a::Union{<:WindowMatrix{T},
#                                   <:Adjoint{T, <:WindowMatrix{T}},
#                                   <:Transpose{T, <:WindowMatrix{T}}};
#     kwargs...) where {T<:Number}
#   return Statistics.mean(identity, a; kwargs...)
# end
# 
# function Statistics.varm(a::Union{<:WindowMatrix{T},
#                                   <:Adjoint{T, <:WindowMatrix{T}},
#                                   <:Transpose{T, <:WindowMatrix{T}}},
#     m::AbstractArray{<:Number};
#     corrected::Bool = true, dims::Dims = :) where {T<:Number}
#   Dims <: Colon || isempty(dims) && error("Not implemented.")
# 
#   # Note: This still has some numerical problems. Consider a call with `dims =
#   # 1` and radically differing values in `m`. Then the `shift` is a bad guess.
#   shift = Statistics.mean(m)
#   r = Statistics.mean(x -> abs2(x - shift), a; dims = dims)
#   r .= max.(r .- abs2.(m .- shift), zero(eltype(r)))
# 
#   if corrected
#     ds = _canonicalize_dims(AbstractMatrix, dims)
#     n = prod([size(a, d) for d in ds])
#     r .*= eltype(r)(n // (n - 1))
#   end
# 
#   return r
# end

"Non-modifying catch-all method for `materialize` (so that it can be called on
array types that are already materialized)."
materialize(a::AbstractArray) = a

# TODO: Should I change the name of `materialize` to `Base.collect`? Does
# `collect` already call `copyto!` by itself?
"Converts an `AbstractWindowArray` into a dense `Array`."
function materialize(a::AbstractWindowArray{T, N}) where {T, N}
  a0 = Array{T, N}(undef, size(a))
  return copyto!(a0, a)
end

# TODO: Materialization/`copyto!` on adjoints/transposes could be done faster
function materialize(a::Transpose{T, <:WindowMatrix{T}}) where {T}
  return transpose(materialize(a.parent))
end

function materialize(a::Adjoint{T, <:WindowMatrix{T}}) where {T}
  return materialize(a.parent)'
end

function compact!(v1::AbstractArray{T},   i1::AbstractArray{Int},
                  v0::AbstractArray{<:T}, i0::AbstractArray{Int},
    k::Int) where {T}
  n1, n0 = 0, 0
  for (j, i) in enumerate(i0)
    l = min(k, i + k - n0 - 1)
    r0 = i .+ (k - l : k - 1)
    r1 = n1 .+ (1:l)
    v1[r1] .= view(v0, r0)
    i1[j] = r1.stop - k + 1
    n1, n0 = r1.stop, r0.stop
  end
  return n1
end

"`compact`s the `WindowMatrix` in-place and returns a new `WindowMatrix`,
either `resize!`ing or employing a `view` of the original data array."
function compact!(a::WindowMatrix; is_sorted::Bool = issorted(a.i),
    crop_only::Bool = false)
  if crop_only
    error("Not implemented.") # TODO
  end
  # TODO: `!is_sorted` case could probably be done more efficiently
  if !is_sorted
    is = sortperm(a.i)
    a.i .= a.i[is]
  end
  n = compact!(a.v, a.i, a.v, a.i, a.k)
  if !is_sorted
    a.i .= a.i[invperm(is)]
  end
  _v = a.v isa Array ? resize!(reshape(a.v, :), n) : view(a.v, Base.OneTo(n))
  return WindowMatrix(_v, a.i, a.k)
end

"Creates a `WindowMatrix` with the most compact memory representation."
function compact(a::WindowMatrix{T},
    ::Val{use_struct_vector} = Val(T <: Complex);
    is_sorted::Bool = issorted(a.i), crop_only::Bool = false
    ) where {T, use_struct_vector}
  V = (use_struct_vector ? StructVector : Vector){T}
  if crop_only
    i0, i1 = a.i[1], a.i[end] + a.k - 1
    v = V(undef, i1 - i0 + 1)
    v .= view(a.v, i0:i1)
    return WindowMatrix(v, a.i .- (i0 - 1), a.k)
  end
  if !is_sorted
    error("Not implemented.") # TODO
  end
  n = 0
  i_prev = -a.k
  for i in a.i
    n += min(a.k, i - i_prev)
    i_prev = i
  end
  v = V(undef, n)
  i = Vector{Int}(undef, length(a.i))
  compact!(v, i, a.v, a.i, a.k)
  return WindowMatrix(v, i, a.k)
end

"Wrapper type to add a slice of ones to the last dimension of an
`AbstractArray`"
struct BiasArray{T, N, P <: AbstractArray} <: AbstractArray{T, N}
  parent::P
  # TODO: Is this constructor okay (since it may create an object of another
  # type) or should I define a function `biasarray` instead?
  function BiasArray(p::P, ::Val{materialized} = Val(false)
      ) where {P <: AbstractArray, materialized}
    a = new{eltype(p), ndims(p), P}(p)
    return materialized ? materialize(a) : a
  end
end

# Override default behavior of constructor for some parent types
BiasArray(p::Union{Array, StructArray, <:AbstractSparseArray}) =
  BiasArray(p, Val(true))

"Wrapper type to add a column of ones to the last dimension of an
`AbstractMatrix`"
const BiasMatrix{T} = BiasArray{T, 2}

Base.size(a::BiasArray) =
  (size(a.parent)[1 : end - 1]..., size(a.parent)[end] + 1)
Base.eltype(::Type{BiasArray{T}}) where {T} = T

Base.IndexStyle(::Type{BiasArray{<:Any, <:Any, P}}) where {P} =
  Base.IndexStyle(P)

function Base.similar(a::BiasArray, ::Type{T}, dims::Dims) where {T}
  p = Base.similar(a.parent, T, (dims[1 : end - 1]..., dims[end] - 1))
  return BiasArray(p; materialized = false)
end

function Base.getindex(a::BiasArray{T}, i::Int) where {T}
  return i ≤ lastindex(a.parent) ? Base.getindex(a.parent, i) : one(T)
end

function Base.getindex(a::BiasArray{T, N}, is::Vararg{Int, N}) where {T, N}
  return is[end] ≤ size(a.parent)[end] ?
    Base.getindex(a.parent, is...) : one(T)
end

function Base.getindex(a::BiasArray{T, N}, is::Vararg{<:Any, N}) where {T, N}
  i = is[end]
  if i isa Colon
    return BiasArray(a.parent[is[1 : end - 1]..., :]; materialized = false)
  elseif i isa Integer
    if i ≤ size(a.parent)[end]
      return a.parent[is...]
    else
      s = size(view(a.parent, is[1 : end - 1]..., firstindex(a.parent, N)))
      if isempty(s)
        return one(T)
      else
        return ones(T, s)
      end
    end
  end
  error("Not implemented.")
end

function Base.setindex!(a::BiasArray, v, i::Int)
  return Base.setindex!(a.parent, v, i)
end

function Base.setindex!(a::BiasArray{T, N}, v, is::Vararg{Int, N}) where {T, N}
  return Base.setindex!(a.parent, v, is...)
end

# TODO: Think about `materialize`, `collect`, `copyto!`, …
"Converts a `BiasArray{T, N, P}` into a `P`. Returns an `Array` if `P <:
AbstractWindowArray`."
function materialize(a::BiasArray{T, N, <:Union{Array, <:AbstractSparseArray}}
    ) where {T, N}
  ns = (size(a)[1 : end - 1]..., 1)
  return cat(a.parent, ones(T, ns); dims = N)
end

function materialize(a::BiasArray{T, N, StructArray}) where {T, N}
  ns = (size(a)[1 : end - 1]..., 1)
  return cat(a.parent, StructArray(ones(T, ns)); dims = N)
end

function materialize(a::BiasArray{T, N, <:AbstractWindowArray}) where {T, N}
  y = Array{T, N}(undef, size(a))
  colons = [(:) for _ in 1 : N - 1]
  view(y, colons..., 1 : size(y)[end] - 1) .= a.parent
  view(y, colons..., size(y)[end]) .= one(T)
  return y
end

@inline function LinearAlgebra.mul!(
    c::AbstractVector, a::BiasMatrix, b::AbstractVector, α::Number, β::Number)
  LinearAlgebra.mul!(c, a.parent, view(b, firstindex(b) : lastindex(b) - 1),
                     α, β)
  c .+= b[end] * α
  return c
end

@inline function LinearAlgebra.mul!(
    c::AbstractVector, a::Adjoint{<:Any, <:BiasMatrix}, b::AbstractVector,
    α::Number, β::Number)
  LinearAlgebra.mul!(view(c, firstindex(c) : lastindex(c) - 1),
                     a.parent.parent', b, α, β)
  c[end] = sum(b) * α + c[end] * β
  return c
end

@inline function LinearAlgebra.mul!(
    c::AbstractVector, a::Transpose{<:Any, <:BiasMatrix}, b::AbstractVector,
    α::Number, β::Number)
  LinearAlgebra.mul!(view(c, firstindex(c) : lastindex(c) - 1),
                     Transpose(a.parent.parent), b, α, β)
  c[end] = sum(b) * α + c[end] * β
  return c
end

# # For performance testing
# export a, a0, x0, x1, ai, a0i, x0i, x1i
# LinearAlgebra.BLAS._set_num_threads(1)
# n = 100000
# k = 250
# v = rand(Float32, n) .* 2 .- 1
# i = 1:5:(n - 2k)
# i = i + vec(sum(rand(Bool, (length(i), 5)); dims = 2))
# a = WindowMatrix(v, i, k)
# a0 = materialize(a)
# x0 = rand(Float32, size(a)[1]) .* 2 .- 1
# x1 = rand(Float32, k) .* 2 .- 1

# vi = rand(ComplexF32, n) .* 2 .- (1 + im)
# ai = WindowMatrix(vi, i, k)
# a0i = materialize(ai)
# x0i = rand(ComplexF32, size(ai)[1]) .* 2 .- (1 + im)
# x1i = rand(ComplexF32, k) .* 2 .- (1 + im)

# export vc, v, a, avc, x0, x1, c0, c1, am, x0r, x1r, c0r, c1r, ar, avcr, amr
# LinearAlgebra.BLAS._set_num_threads(1)
# number_frames = 30 * 15 * 2 * 50
# number_channels = 32
# samples_per_frame = 5
# window_size = 50
# vc = Vector(rand(ComplexF32, number_frames * number_channels * samples_per_frame
#                  + window_size * number_channels * 2) * 2 .- (1+im));
# v = StructVector(vc);
# a = WindowMatrix(v, (collect(1:number_frames)) * number_channels *
#                  samples_per_frame, window_size * number_channels);
# avc = WindowMatrix(vc, (collect(1:number_frames)) * number_channels *
#                    samples_per_frame, window_size * number_channels);
# x0 = rand(ComplexF32, number_frames);
# x1 = rand(ComplexF32, window_size * number_channels);
# c0 = zeros(ComplexF32, number_frames);
# c1 = zeros(ComplexF32, window_size * number_channels);
# am = materialize(a);
# x0r = real(x0);
# x1r = real(x1);
# c0r = real(c0);
# c1r = real(c1);
# ar = real(a);
# avcr = real(avc);
# amr = real(am);
# a = mul_prepare(a; flags = MEASURE);
# avc = mul_prepare(avc);

end
