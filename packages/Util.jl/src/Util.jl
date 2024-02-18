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
  # shift_range,
  # pad_range,
  # find_true_ranges,
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

# function shift_range(range::UnitRange{<:Number}, left::Number, right::Number)
#   (range.start+left):(range.stop+right)
# end

# function shift_range(range::UnitRange{<:Number}, shift::Number)
#   shift_range(range, shift, shift)
# end

# function pad_range(range::UnitRange{<:Number}, left::Number, right::Number)
#   (range.start-left):(range.stop+right)
# end

# function pad_range(range::UnitRange{<:Number}, padding::Number)
#   pad_range(range, padding, padding)
# end

# function find_true_ranges(ys::AbstractVector{Bool})
#   delimiters = ys - circshift(ys, 1)
#   map(:, findall(delimiters .== 1), findall(delimiters .== -1) .- 1)
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

function struct_equality(x::T, y::T) where {T}
  return all([getfield(x, fn) == getfield(y, fn) for fn in fieldnames(T)])
end

end

