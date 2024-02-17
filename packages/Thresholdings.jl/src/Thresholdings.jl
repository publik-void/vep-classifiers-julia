"Thresholding functions and machine learning models"
module Thresholdings

using DataProcessors: DataProcessor, fit, apply
using Optim: optimize, GoldenSection
import DataProcessors

export
  t2,
  t3,
  t2a,
  t3a,
  Thresholding,
  fit,
  apply,
  Binary,
  Ternary

"""
    t2(z, t0, y0 = 0, y1 = 1, ϕ = 0)

Binary thresholding function.

Returns `y0` if `z ≤ t0` and `y1` otherwise. In the complex case, the angle `ϕ`
effectively rotates the partitioning of the complex plane by `t0`.
"""
@inline function t2(x::Real, t0::Real,
    y0::T = 0, y1::T = 1,
    ϕ::Real = 0) where {T}
  return x ≤ t0 ? y0 : y1
end

@inline function t2(z::Complex, t0::Real,
    y0::T = 0, y1::T = 1,
    ϕ::Real = 0) where {T}
  y, x = real(z), imag(z)
  slope = tan(ϕ * .5π)
  return y - x * slope ≤ t0 ? y0 : y1
end

# TODO: Complete the documentation, including the complex case.
"""
    t3(z, t0, t1, y0 = -1, y1 = 0, y2 = 1, ϕ = 0, e⁻¹ = 1)

Ternary thresholding function.
"""
@inline function t3(x::Real, t0::Real, t1::Real,
    y0::T = -1, y1::T = 0, y2::T = 1,
    ϕ::Real = 0, e⁻¹::Real = 1) where {T}
  return x < t0 ? y0 : (x ≤ t1 ? y1 : y2)
end

@inline function t3(z::Complex, t0::Real, t1::Real,
    y0::T = -1, y1::T = 0, y2::T = 1,
    ϕ::Real = 0, e⁻¹::Real = 1) where {T}
  y, x = real(z), imag(z)
  c, r = 1//2 * (t0 + t1), 1//2 * (t1 - t0)
  slope = tan(ϕ * .5π)
  return e⁻¹^2 * x^2 + (y - c)^2 ≤ r^2 ? y1 :
    (y - x * slope ≤ c ? y0 : y2)
end

"Returns the fraction of elements in `y` which match the elements in `b` when
thresholded by `t2`."
function t2a(y::AbstractArray, b::AbstractArray{T}, t0::Real,
    y0::T = 0, y1::T = 1,
    ϕ::Real = 0) where {T}
  acc = 0
  for (y, b) in zip(y, b)
    acc += t2(y, t0, y0, y1, ϕ) == b
  end
  return acc // length(y)
end

"Returns the fraction of elements in `y` which match the elements in `b` when
thresholded by `t3`."
function t3a(y::AbstractArray, b::AbstractArray{T}, t0::Real, t1::Real,
    y0::T = -1, y1::T = 0, y2::T = 1,
    ϕ::Real = 0, e⁻¹::Real = 1) where {T}
  acc = 0
  for (y, b) in zip(y, b)
    acc += t3(y, t0, t1, y0, y1, y2, ϕ, e⁻¹) == b
  end
  return acc // length(y)
end

"Abstract supertype for hyperparameters of a thresholding model."
abstract type Thresholding <: DataProcessor end

# TODO: For the `fit` methods, I use a general 1D optimizer at the moment.
# Threshold optimization can probably be done with a better algorithm,
# especially for the multivariate case.

"2-class thresholding model specification"
struct Binary <: Thresholding
  t0::Real
end

function DataProcessors.fit(
    ::Type{Binary},
    y::AbstractVector{<:Number},
    b::AbstractVector{<:T},
    classes::AbstractVector{<:T}) where {T<:Any}
  y0, y1 = classes
  t0_min, t0_max = extrema(real, y)
  obj(t0) = 1. - t2a(y, b, t0, y0, y1)
  t0 = optimize(obj, t0_min, t0_max, GoldenSection()).minimizer
  return Binary(t0)
end

function DataProcessors.apply(
    t::Binary,
    y::AbstractVector{<:Number},
    classes::AbstractVector{<:Any})
  y0, y1 = classes
  return t2.(y, t.t0, y0, y1)
end

"3-class thresholding model specification"
struct Ternary <: Thresholding
  t0::Real
  t1::Real
end

function DataProcessors.fit(
    ::Type{Ternary},
    y::AbstractVector{<:Number},
    b::AbstractVector{<:T},
    classes::AbstractVector{<:T}) where {T<:Any}
  y0, y1, y2 = classes
  t0_min, t0_max = extrema(real, y)
  t1_max = t0_max
  obj(t0, t1) = 1. - t3a(y, b, t0, t1, y0, y1, y2)
  t0 = optimize(t0 -> obj(t0, optimize(t1 -> obj(t0, t1), t0, t1_max,
    GoldenSection()).minimizer), t0_min, t0_max, GoldenSection()).minimizer
  t1 = optimize(t1 -> obj(t0, t1), t0, t1_max, GoldenSection()).minimizer
  return Ternary(t0, t1)
end

@inline function DataProcessors.apply(
    t::Ternary,
    y::AbstractVector{<:Number},
    classes::AbstractVector{<:Any})
  y0, y1, y2 = classes
  return t3.(y, t.t0, t.t1, y0, y1, y2)
end

end
