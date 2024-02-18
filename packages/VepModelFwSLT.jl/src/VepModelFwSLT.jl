module VepModelFwSLT

using WindowArrays: WindowMatrix, BiasMatrix, BiasArray, mul_prepare,
  materialize, gfpv_new_plan
using Standardizations: AffStd, fit, apply!
using Thresholdings: Thresholding, fit, apply
using VepModels: VepModel, ab, fit, apply, window_signal,
  label_markers_to_window_indexes
using IterativeSolvers: lsmr!
import VepModels

export
  FwSLT,
  ab,
  fit,
  apply

"""
    FwSLT(window_size, signal_offset, λ, std_mode, bias, thresholding, \
      classes, lsmr_kw)
Frame-wise sliding window model consisting of an affine standardization, a ridge
regression, and thresholding. Tries to make use of `WindowMatrix`.

# Arguments

* `window_size`: Length of sliding windows in seconds.
* `signal_offset`: Window start offset relative to markers.
* `λ`: Regularization weight for the regression.
* `std_mode`: `Symbol` specifying the feature standardization mode for an \
  `AffStd`. See documentation for `DataProcessors.fit(AffStd, …)`.
* `bias`: `Bool` specifying whether to include a bias term in the regression.
* `thresholding`: A subtype of `Thresholdings.Thresholding` to indicate what
  kind of thresholding to use.
* `classes`: An `AbstractVector` containing one assigned output value for each
  thresholding class.
* `label_thresholds`: An `AbstractVector` containing the thresholds to be used
  to convert training labels to thresholding classes.
* `lsmr_kw`: Keyword arguments for the LSMR algorithm (e.g. stopping \
  tolerances). See documentation for `IterativeSolvers.lsmr!`.
"""
struct FwSLT <: VepModel
  window_size::Real
  signal_offset::Real
  λ::Real
  std_mode::Symbol
  bias::Bool
  thresholding::Type{<:Thresholding}
  classes::AbstractVector
  label_thresholds::AbstractVector{<:Number}
  lsmr_kw
end

"A wrapper for `mul_prepare` that handles non-`WindowMatrix` matrices and prints
a notification in case no FFT plan was precomputed."
_mul_prepare(a::AbstractMatrix{<:Number}) = a
_mul_prepare(a::WindowMatrix{<:Number}) =
  mul_prepare(a; gfpv_flags = gfpv_new_plan)
_mul_prepare(a::BiasMatrix{<:Number}) =
  BiasArray(_mul_prepare(a.parent); materialized = false)

"""
    ab(model::FwSLT, signal, label_markers, sampling_rate; kw...)

Creates a feature matrix `a` of sliding windows and a corresponding label vector
`b` from the input `signal` and input `label_markers`. Returns `(a, b)`.

`signal` is an ``n×m`` matrix with ``n`` time points and ``m`` channels.

`label_markers` specifies labels (any `Number`s unequal to `sentinel`) and
corresponding time points where windows should be placed. See
`VepModels.label_markers_to_window_indexes`. If `keys(label_markers)` is not
already sorted in ascending order, the keyword argument `is_sorted` should be
set to `false`.

The `sampling_rate` of the signal is measured in Hz (time points per second).

Further keyword arguments `kw...` are passed on to `VepModels.window_signal`.
"""
function VepModels.ab(model::FwSLT, signal::AbstractMatrix{<:Number},
    label_markers, sampling_rate::Real; sentinel = nothing, kw...)
  window_size = Int(round(model.window_size * sampling_rate))
  signal_offset = Int(round(model.signal_offset * sampling_rate))

  is = label_markers_to_window_indexes(label_markers; sentinel)
  b = [label_markers[i] for i in is]
  a = window_signal(signal, is, window_size, signal_offset; kw...)

  if model.bias; a = BiasArray(a; materialized = false); end

  @assert size(a, 1) == length(b) "`FwSLT`: `size(a, 1)` is $(size(a, 1)), " *
    "but `length(b)` is $(length(b))"
  return a, b
end

"""
    fit(model::FwSLT, a, b; allow_overwrite_a = true)

Trains the `model` on the feature matrix `a` with label vector `b` and returns
the resulting fit `(; std, x, t)`.

Pass `allow_overwrite_a = false` if the data of `a` should remain unmodified.
"""
function VepModels.fit(model::FwSLT, a::AbstractMatrix{<:Number},
    b::AbstractVector{<:Number}; allow_overwrite_a::Bool = true)
  a = _mul_prepare(a)

  std = fit(AffStd, a, model.std_mode == () ? :identity : model.std_mode)
  apply_std_kw = allow_overwrite_a ? () : (; lazy = true)
  a = apply!(std, a; apply_std_kw...)

  x = rand(eltype(a), size(a, 2)) .* 4 .- 2 .- (eltype(a) <: Complex ? 2im : 0)
  # tic = time()
  lsmr!(x, a, b; λ = model.λ, model.lsmr_kw...)
  # toc = time()
  # printstyled("LSMR took $(toc-tic)s for $(typeof(a)) of size $(size(a))";
  #   color = 12); println("")

  b_thresholded = apply(model.thresholding(model.label_thresholds...), b,
    model.classes)
  t = fit(model.thresholding, a * x, b_thresholded, model.classes)
  return (; std, x, t)
end

"""
    apply(model::FwSLT, a, fit; allow_overwrite_a = true)

Applies the `fit` (as returned by `fit(model, …)`) to the feature matrix `a` and
returns the prediction vector.

Pass `allow_overwrite_a = false` if the data of `a` should remain unmodified.
"""
function VepModels.apply(model::FwSLT, a::AbstractMatrix{<:Number},
    fit::NamedTuple{(:std, :x, :t), Tuple{AffStd, AbstractVector{<:Number},
      Thresholding}};
    allow_overwrite_a::Bool = true)
  a = _mul_prepare(a)
  apply_std_kw = allow_overwrite_a ? () : (; lazy = true)
  a = apply!(fit.std, a; apply_std_kw...)
  return apply(fit.t, a * fit.x, model.classes)
end

end
