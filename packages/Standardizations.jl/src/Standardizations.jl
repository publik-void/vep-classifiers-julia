"Feature standardization on feature matrices"
module Standardizations

using DataProcessors: DataProcessor, fit, apply, apply!, is_id
using WindowArrays: WindowMatrix, BiasMatrix, BiasArray
using Statistics: mean, std
using LinearAlgebra: Adjoint, Transpose
import DataProcessors
import LinearAlgebra

export
  Standardization,
  AffStd,
  AffStdMatrix,
  fit,
  apply,
  apply!,
  is_id

"Abstract supertype for feature standardization parameters"
abstract type Standardization <: DataProcessor end

"Holds parameters to standardize features through affine transformation."
struct AffStd{M<:Union{Nothing, <:AbstractMatrix{<:Number}},
              S<:Union{Nothing, <:AbstractMatrix{<:Number}}} <: Standardization
  m::M
  s::S
end

function DataProcessors.is_id(std::AffStd; full_check = false)
  if isnothing(std.m) && isnothing(std.s)
    return true
  elseif full_check
    return (isnothing(std.m) || all(std.m .== false)) &&
           (isnothing(std.s) || all(std.s .== true))
  else
    return false
  end
end

# TODO: Think about using `sr` and `msr` in `AffStd` directly and then use that
# `AffStd` in `AffStdMatrix`.

"Wraps an `AbstractMatrix` with an `AffStd` applied to it lazily."
struct AffStdMatrix{T<:Number,
    A<:AbstractMatrix{T},
    SR<:Union{Nothing, <:AbstractArray{<:Number}},
    MSR<:Union{Nothing, <:AbstractArray{<:Number}}} <: AbstractMatrix{T}
  parent::A
  sr::SR # Reciprocal of `AffStd(…).s`
  msr::MSR # `AffStd(…).m .* sr`
end

@inline function default_use_lazy(a::AbstractMatrix)
  a isa WindowMatrix && return true
  return false
end

function AffStdMatrix(a::AbstractMatrix{<:Number}, std::AffStd;
    materialized = !default_use_lazy(a))
  # TODO: Does this allocate more than necessary?
  # TODO: Perhaps canonicalize this to always create vectors (`vec` may prduce
  # something like a `reshape`), or would it be nicer not to do that?
  materialized && return apply(a, std)
  has_m, has_s = !isnothing(std.m), !isnothing(std.s)
  sr = !has_s ? nothing : vec(inv.(transpose(std.s)))
  msr = !has_m ?
    nothing : vec((!has_s ? transpose(std.m) : transpose(std.m) .* sr))
  return AffStdMatrix(a, sr, msr)
end

Base.size(a::AffStdMatrix) = size(a.parent)
Base.eltype(::Type{AffStdMatrix{T}}) where {T} = T

@inline function Base.getindex(a::AffStdMatrix{<:Number, <:Any, SR, MSR},
    i0::Int, i1::Int) where {SR, MSR}
  has_s, has_m = !(SR <: Nothing), !(MSR <: Nothing)
  !has_s && !has_m && return getindex(a.parent, i0, i1)
  has_s  && !has_m && return getindex(a.parent, i0, i1) * a.sr[i1]
  !has_s &&  has_m && return getindex(a.parent, i0, i1) - a.msr[i1]
  has_s  &&  has_m && return getindex(a.parent, i0, i1) * a.sr[i1] - a.msr[i1]
end

# TODO: Implement `copyto!` for `AffStdMatrix`.

@inline function LinearAlgebra.mul!(
    c::AbstractVector{<:Number},
    a::AffStdMatrix{<:Number, <:Any, SR, MSR},
    b::AbstractVector{<:Number},
    α::Number, β::Number) where {SR, MSR}
  has_s, has_m = !(SR <: Nothing), !(MSR <: Nothing)
  !has_s && !has_m && return LinearAlgebra.mul!(c, a.parent, b, α, β)
  has_s  && !has_m && return LinearAlgebra.mul!(c, a.parent, b .* a.sr, α, β)
  !has_s && has_m  && (LinearAlgebra.mul!(c, a.parent, b, α, β);
                       c .-= transpose(b) * a.msr .* α;
                       return c)
  has_s  && has_m  && (LinearAlgebra.mul!(c, a.parent, b .* a.sr, α, β);
                       c .-= transpose(b) * a.msr .* α;
                       return c)
end

@inline function LinearAlgebra.mul!(
    c::AbstractVector{<:Number},
    a::Adjoint{<:Number, <:AffStdMatrix{<:Number, <:Any, SR, MSR}},
    b::AbstractVector{<:Number},
    α::Number, β::Number) where {SR, MSR}
  has_s, has_m = !(SR <: Nothing), !(MSR <: Nothing)
  !has_s && !has_m && return LinearAlgebra.mul!(c, a.parent.parent', b, α, β)
  # TODO: Other cases
  # TODO: Profile and optimize this.
  has_s  && has_m  && return (c .= ((a.parent.parent' * b) .*
    conj.(a.parent.sr) .- conj.(a.parent.msr) .* sum(b)) .* α .+ c .* β;
                              return c)
end

@inline function DataProcessors.apply!(std::AffStd,
    y::AbstractMatrix{<:Number}, a::AbstractMatrix{<:Number})
  has_m, has_s = !isnothing(std.m), !isnothing(std.s)
  if has_m
    if has_s; y .= (a .- std.m) ./ std.s
    else;     y .= a .- std.m
    end
  else
    if has_s; y .= a ./ std.s
    else;     y .= a
    end
  end
  return y
end

# Specialization for `BiasMatrix` does not standardize bias term.
@inline function DataProcessors.apply!(std::AffStd,
    y::BiasMatrix{<:Number}, a::BiasMatrix{<:Number})
  apply!(std, y.parent, a.parent)
  return y
end

@inline function DataProcessors.apply!(std::AffStd,
    a::AbstractMatrix{<:Number}; lazy = default_use_lazy(a))
  has_m, has_s = !isnothing(std.m), !isnothing(std.s)
  !has_m && !has_s && return a
  lazy && return apply(std, a; lazy = true)
  # Specialization for `WindowMatrix` auto-materializes when `!lazy`. Note:
  # using `similar(a)` here is more efficient than calling `materialize(a)`.
  a isa WindowMatrix && return apply!(std, similar(a), a)
  return apply!(std, a, a)
end

# Specialization for `BiasMatrix` does not standardize bias term.
# Note: This specialization is only needed to support parent types that are not
# directly supported by `apply!(std, a.parent, a.parent)`.
@inline function DataProcessors.apply!(std::AffStd,
    a::BiasMatrix{<:Number}; kwargs...)
  return BiasArray(apply!(std, a.parent, kwargs...); materialized = false)
end

@inline function DataProcessors.apply(std::AffStd,
    a::AbstractMatrix{<:Number}; lazy = default_use_lazy(a))
  has_m, has_s = !isnothing(std.m), !isnothing(std.s)
  !has_m && !has_s && return a
  !lazy && return apply!(std, similar(a), a)
  return AffStdMatrix(a, std; materialized = false)
end

# Specialization for `BiasMatrix` does not standardize bias term.
@inline function DataProcessors.apply(std::AffStd,
    a::BiasMatrix{<:Number}; kwargs...)
  return BiasArray(apply(std, a.parent; kwargs...); materialized = false)
end

# TODO: Confirm that `fit` runs efficiently on `WindowMatrix`.
"""
    fit(::Type{AffStd}, a, mode = :identity; exclude = nothing)

Create affine standardization parameters fitted to the matrix `a` and return
them as an object of type `AffStd`.

Supported `mode`s:
* `:identity`: No transformation.
* `:μ_absgeom`: Centers to the mean and scales to unity geometric mean of \
  absolute value.
* `:μ_σ`: Centers to the mean and scales to unity variance.

If `exclude` is not `nothing`, it is interpreted as an index (as accepted by \
  `setindex!`, e.g. a `Vector{Int}`) of columns to exclude from standardization.
"""
function DataProcessors.fit(::Type{AffStd}, a::AbstractMatrix{<:Number},
    mode::Symbol = :identity;
    exclude::Union{Nothing, <:AbstractVector{<:Integer}} = nothing)
  m, s = nothing, nothing
  if mode == :identity
    # Nothing to be done

  elseif mode == :μ_absgeom
    m = mean(a; dims = 1)
    t = typeof(abs(a[1]))(1e-18)
    s = exp.(mean(log.(max.(abs.(a .- m), t)); dims = 1)) # TODO: Efficiency

  elseif mode == :μ_σ
    m = mean(a; dims = 1)
    s = std(a; dims = 1, mean = m)
  else
    error("Unrecognized standardization fit mode `:$(mode)`.")
  end
  if !isnothing(exclude)
    !isnothing(m) && (m[exclude] .= zero(eltype(m)))
    !isnothing(s) && (s[exclude] .=  one(eltype(s)))
  end
  return AffStd(m, s)
end

# Specialization for `BiasMatrix` does not fit bias term.
function DataProcessors.fit(::Type{AffStd}, a::BiasMatrix{<:Number},
    mode::Symbol = :identity;
    exclude::Union{Nothing, <:AbstractVector{<:Integer}} = nothing)
  return fit(AffStd, a.parent, mode; exclude = exclude)
end

end
