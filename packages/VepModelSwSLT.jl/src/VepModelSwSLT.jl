"Module providing the `SwSLT` model. See `VepModelSwSLT.SwSLT`."
module VepModelSwSLT

using Thresholdings: Thresholding
using VepModels: VepModel, ab, fit, apply, window_signal
using VepModelFwSLT: FwSLT
import VepModels

export
  SwSLT,
  find_label_thresholds,
  ab,
  fit,
  apply

"""
    SwSLT(segment_length, λ, std_mode, bias, thresholding, classes, \
      label_thresholds = nothing; kw...)

Segment-wise model for stimulation pattern segments of fixed length, consisting
of an affine standardization, a ridge regression, and thresholding.

# Arguments

Internally, the model repurposes `VepModelFwSLT`. Arguments to `SwFLT` for which
there exists an argument to `FwSLT` of the same name fulfill the same role. In
the following, only the arguments unique to `SwSLT` are described:

* `segment_length`: The length of a single segment (in frames, usually, but more
  broadly, in number of lightness values per segment).
* `label_thresholds`: What is unique to `SwSLT` about this argument is that it
  can be omitted (or set to `nothing`). If this is done, `classes` must be a
  sorted `AbstractVector{<:Real}`. Then, the `label_thresholds` will
  automatically be chosen (via `find_label_thresholds`) to lie exactly in the
  middle between each pair of consecutive elements of `classes`.
"""
struct SwSLT{SL <: Integer, FW <: FwSLT} <: VepModel
  segment_length::SL
  fw::FW
end

for (p, f) in ((:(Base.getproperty), :(Base.getfield)),
               (:(Base.setproperty!), :(Base.setfield!))) quote
  function $p(x::SwSLT, name::Symbol, args...)
    name in propertynames(getfield(x, :fw)) &&
      return $p(getfield(x, :fw), name, args...)
    return $f(x, name, args...)
  end
end |> eval end

Base.propertynames(x::SwSLT) =
  unique((fieldnames(SwSLT)..., propertynames(x.fw)...))

SwSLT(segment_length, λ, std_mode, bias, thresholding, classes,
    label_thresholds = nothing; kw...) =
  SwSLT(segment_length, FwSLT(1, 0, λ, std_mode, bias, thresholding, classes,
    @something(label_thresholds, find_label_thresholds(classes)); kw...))

"""
    find_label_thresholds(classes)

Creates a vector of values that lie exactly in the middle between each
consecutive class pair. `issorted(classes)` must be `true`. This is equivalent
to a running mean of size 2 without boundary padding.
"""
function find_label_thresholds(classes::AbstractVector{<:Real})
  if !issorted(classes)
    throw(ArgumentError("`find_label_thresholds`: `classes` must be sorted"));
  end
  return 1//2 .* (
      view(classes, firstindex(classes) : lastindex(classes) - 1) .+
      view(classes, firstindex(classes) + 1 : lastindex(classes)))
end

"""
    ab(model::SwSLT, lightnesses, labels; kw...)

Creates a feature marix `a` of segments in `lightnesses` and returns it with
a copy of the label vector `b = labels`.

`lightnesses` is a vector of length `length(labels) * model.segment_length` that
contains the individual lightness values for one segment after another.
`lightnesses` can also be a matrix, where instead of a single value, each row
(representing a time point) can have multiple values (e.g. RGB values or EEG
channels).

`labels` is a vector of numerical segment labels, where the segments are in the
same order as in `lightnesses`.

Keyword arguments `kw...` are passed on to `VepModelFwSLT`. Defaults are set to
`materialized = true, compact = false`.
"""
function VepModels.ab(model::SwSLT, lightnesses::AbstractVecOrMat{<:Number},
    labels::AbstractVector{<:Number}; materialized = true, compact = false,
    kw...)
  signal = transpose(reshape(transpose(lightnesses),
    (model.segment_length * size(lightnesses, 2), :)))
  size(signal, 1) == length(labels) || throw(DimensionMismatch(
    "`ab`: `length(lightnesses) ÷ model.segment_length ≠ length(labels).`"))

  return ab(model.fw, signal, labels, 1; materialized, compact, kw...);
end

"""
    fit(model::SwSLT, a, b; kw...)

Trains the `model` on the feature matrix `a` with label vector `b` and returns
the resulting fit `(; std, x, t)`.

Further keyword arguments `kw...` will be passed on to `fit(::FwSLT, …)`.
"""
function VepModels.fit(model::SwSLT,
    a::AbstractMatrix{<:Number}, b::AbstractVector{<:Number};
    init_scale = 2 // model.segment_length, kw...)
  return fit(model.fw, a, b; init_scale, kw...)
end

"""
    apply(model::SwSLT, a, fit; kw...)

Applies the `fit` (as returned by `fit(model, …)`) to the feature matrix `a` and
returns the prediction vector.

Further keyword arguments `kw...` will be passed on to `apply(::FwSLT, …)`.
"""
function VepModels.apply(model::SwSLT, a::AbstractMatrix{<:Number}, fit; kw...)
  return apply(model.fw, a, fit; kw...)
end

end
