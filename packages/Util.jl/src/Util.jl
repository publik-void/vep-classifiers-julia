"Miscellaneous utility functions"
module Util

using Base.Iterators: partition, take, drop
using Base.Threads: nthreads

export
  # cis2pi,
  # angle2pi,
  # sin2pi,
  # cos2pi,
  # lerp,
  # delta,
  struct_equality

# function cis2pi(z::T) where {T<:Number}
#   cispi(T(2) * z)
# end

# function angle2pi(z::T) where {T<:Number}
#   angle(z) / real(T(2π))
# end

# function sin2pi(z::T) where {T<:Number}
#   sinpi(T(2) * z)
# end

# function cos2pi(z::T) where {T<:Number}
#   cospi(T(2) * z)
# end

# function lerp(y0, y1, x)
#   return (1 .- x) .* y0 .+ x .* y1
# end

# function lerp(y0, y1, x, x0, x1)
#   return lerp(y0, y1, (x .- x0) ./ (x1 .- x0))
# end

# function delta(v::AbstractVector{T};
#     cyclic = false, inv = false, shift = false) where {T<:Number}
#   n = length(v) - 1
#   ds = Vector{T}(undef, n + cyclic)
#   o = Int(cyclic && !shift)
#   @inbounds @simd for i in 1:n
#     ds[i + o] = !inv ? v[i + 1] - v[i] : v[i] - v[i + 1]
#   end
#   if cyclic
#     ds[!shift ? 1 : end] = !inv ? v[1] - v[end] : v[end] - v[1]
#   end
#   return ds
# end

function struct_equality(x::T, y::T, comp = (==)) where {T}
  if isstructtype(T)
    # Don't recurse, rely on `comp` definition for field types
    return all(comp(getfield(x, fn), getfield(y, fn)) for fn in fieldnames(T))
  else
    return comp(x, y)
  end
end

end

