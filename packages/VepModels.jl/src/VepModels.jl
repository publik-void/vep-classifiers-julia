"General abstractions for VEP classification models making use of `ab`, `fit`,
and `apply` methods."
module VepModels

using Util: struct_equality
using DataProcessors: DataProcessor, fit, apply
using WindowArrays: WindowMatrix, compact!, materialize
using Random: randperm
import WindowArrays

export
  VepModel,
  ab,
  fit,
  apply,
  cv_splits,
  align_with_tick,
  find_ticks,
  label_markers_to_window_indexes,
  label_markers_to_window_indexes_and_labels,
  window_signal

"Supertype for hyperparameters for turning the input signals and labels into a
matrix of sliding windows and a label vector (`ab`), training (`fit`) and
prediction (`apply`)."
abstract type VepModel <: DataProcessor end

Base.:(==)(x::M, y::M) where {M<:VepModel} = struct_equality(x, y)

# function Base.show(io::IO, m::VepModel)
#   print(io, typeof(m))
#   if fieldcount(typeof(m)) > 0
#     ps = join(["$(f)=$(getfield(m, f))" for f in fieldnames(typeof(m))], " ")
#     print(io, "<$(ps)>")
#   end
# end

"Creates a feature matrix and label vector for a model from the input data."
function ab end

"Creates a set of cross-validation splits and the permutation needed for
reconstructing the original ordering of the samples."
function cv_splits(n::Integer; folds::Integer = 6, splitting::Symbol = :shuffle)
  is = Int.(round.(range(1, n + 1, length = folds + 1)))
  if splitting == :even
    splits = [(vcat(1 : i0 - 1, i1:n), i0 : i1 - 1)
              for (i0, i1) in zip(is[1 : end - 1], is[2:end])]
    deperm = nothing
  elseif splitting == :shuffle
    # TODO: Create reproducability? Not completely trivial when threading…
    perm = randperm(n)
    splits = [(vcat(perm[1 : i0 - 1], perm[i1:n]), perm[i0 : i1 - 1])
              for (i0, i1) in zip(is[1 : end - 1], is[2:end])]
    deperm = invperm(perm)
  else
    error("`cv_splits`: unrecognized `splitting = :$(splitting)`.")
  end
  return splits, deperm
end

"""
    align_with_tick(index, ticks, tie_to_left = true)

Aligns the `index` with the nearest value change in `ticks`.
"""
function align_with_tick(index::Integer, ticks::AbstractVector{<:Any},
    tie_to_left::Bool = true)
  i0 = findprev(s -> s ≠ ticks[index], ticks, index)
  i1 = findnext(s -> s ≠ ticks[index], ticks, index)
  if isnothing(i0)
    i0 = 0
  end
  if isnothing(i1)
    i1 = length(ticks)
  end
  i0 += 1
  return (tie_to_left ? (≤) : (<))(index - i0, i1 - index) ? i0 : i1
end

"""
find_ticks(ticks, from = 1, to = length(ticks))

Returns a vector of indexes in the range `from:to` where the value of `ticks` is
different from the previous value.
"""
function find_ticks(ticks::AbstractVector,
    from::Int = 1, to::Int = length(ticks))
  n = 0
  last = ticks[end]
  for (i, x) in enumerate(view(ticks, Base.OneTo(to)))
    if x ≠ last && from ≤ i
      n += 1
    end
    last = x
  end
  is = Vector{Int}(undef, n)
  n = 0
  last = ticks[end]
  for (i, x) in enumerate(view(ticks, Base.OneTo(to)))
    if x ≠ last && from ≤ i
      n += 1
      is[n] = i
    end
    last = x
  end
  return is
end

"""
    label_markers_to_window_indexes(label_markers, sentinel = nothing)

Creates an iterator over indexes from `label_markers` where the label value is
not equal to `sentinel`. Preserves ordering of `keys(label_markers)`.

`label_markers` can be any object that supports the `keys`, `values`, and
`valtype` methods. The keys should be of type `T where {T <: Integer}`. The
values should be of type `T where {T <: Union{typeof(sentinel), Number}}`. A
value that is unequal to `sentinel` indicates that a window should be placed
around the time point (array index between ``1`` and ``n``) indicated by the
key.

`label_markers` could e.g. be a `Dict{Int, LabelType}` (sparse representation)
or a `Vector{Union{typeof(sentinel), LabelType}}` (dense representation where
any time point without an associated label is designated by `sentinel`).
"""
function label_markers_to_window_indexes(label_markers, sentinel = nothing,
    ::Val{return_labels} = Val(false)) where {return_labels}
  if return_labels; vs = values(label_markers); end
  if isnothing(sentinel) &&
      !(Nothing <: (return_labels ? eltype(vs) : valtype(label_markers)))
    ks = keys(label_markers)
    is = ks isa LinearIndices{1} ? first(ks.indices) : ks
  else
    is = (k for (k, v) in
      zip(keys(label_markers), return_labels ? vs : values(label_markers))
      if v ≠ sentinel)
    vs = (v for v in vs if v ≠ sentinel)
  end
  return return_labels ?
    (is, reshape(vs isa AbstractArray ? vs : collect(vs), (:,))) : is
end

"""
    label_markers_to_window_indexes_and_labels(args...; kw...)

Like `label_markers_to_window_indexes`, but also return a vector of labels.
"""
label_markers_to_window_indexes_and_labels(args...; kw...) =
  label_markers_to_window_indexes(args..., Val(true); kw...)

"""
    window_signal(signal, is, window_size, signal_offset, \
      materialized = Val(false), compact = Val(true), copy = compact, \
      copy_is = Val(true); is_sorted = true, kw...)

Creates a `WindowMatrix` of (potentially overlapping) windows into the `signal`
(a ``n×m`` matrix of ``n`` time points and ``m`` channels). `is` is an iterator
of element type `<:Integer` over time point indexes where windows should be
created. For any `i ∈ is`, the window will start at `i + signal_offset` and have
a length of `window_size`. `window_size` and `signal_offset` are measured in
number of time points.

If `materialized` is `Val(true)`, no `WindowArrays.WindowMatrix` will be
utilized.

If `compact` is `Val(true)`, the underlying data will be rewritten to discard
all time points that are not covered by a window. This will e.g. speed up matrix
multiplications in cases where substantial time spans of the signal are
irrelevant. If `copy` is `Val(true)`, a copy of the underlying signal matrix
will be made, so that the original data will not be overwritten.

If `is` is an `AbstractArray`, it will always be overwritten unless `copy_is` is
`Val(true)`, in which case a copy of its data will be made.

If `is` is not already sorted in ascending order, `is_sorted` should be set to
`false`.

Any further keyword arguments will be passed on to `WindowArrays.compact` or
`WindowArrays.compact!`.
"""
function window_signal(signal::AbstractMatrix, is, window_size::Integer,
    signal_offset::Integer, ::Val{materialized} = Val(false),
    val_compact::Val{compact} = Val(true), ::Val{copy} = val_compact,
    ::Val{copy_is} = Val(!(is isa Array)); is_sorted::Bool = true, kw...
    ) where {materialized, compact, copy, copy_is}
  m = size(signal, 2)

  if !copy_is && is isa AbstractArray
    is .= (is .- 1 .+ signal_offset) .* m .+ 1; _is = is
  elseif is isa AbstractArray
    _is = (is .- 1 .+ signal_offset) .* m .+ 1
  else
    _is = [(i - 1 + signal_offset) * m + 1 for i in is]
  end

  _compact = compact && !materialized
  _copy = copy && !materialized

  _signal = (_copy && !_compact) ? deepcopy(signal) : signal
  a = WindowMatrix(transpose(_signal), _is, window_size * m)

  _a = _compact ?
    (_copy ? WindowArrays.compact : compact!)(a; is_sorted, kw...) : a

  return materialized ? materialize(_a) : _a
end

end
