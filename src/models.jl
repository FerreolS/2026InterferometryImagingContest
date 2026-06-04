abstract type OIModel end

Base.@kwdef struct OIDataPoint{U, V, L, P, T}
    u::U
    v::V
    λ::L = nothing
    p::P = nothing
    t::T = nothing
end

(a::AbstractVector{T})(::OIDataPoint) where {T <: Number} = a

struct Punct{P, F} <: OIModel
    position::P
    flux::F
end

function ((; position, flux)::Punct)(d::OIDataPoint)
    x, y = position(d)
    flux = get_flux(flux, d)
    (; u, v) = d
    return flux * cis(2π * (u * x .+ v * y))
end

function (A::NTuple{N, OIModel})(d::OIDataPoint) where {N}
    return mapreduce(x -> x(d), +, A; init = zero(ComplexF64))
end

function get_flux(A::NTuple{N, OIModel}, d) where {N}
    return mapreduce(x -> get_flux(x, d), +, A)
end

get_flux((; flux)::OIModel, d::OIDataPoint) = get_flux(flux, d)
get_flux((; flux)::OIModel, λ::Number) = get_flux(flux, λ)
get_flux(flux::F, ::OIDataPoint) where {F <: Number} = flux
get_flux(flux::AbstractArray{T}, (; λ)::OIDataPoint{U, V, T2, Nothing, Nothing}) where {U, V, T <: Number, T2 <: Number} = length(flux) == 1 ? flux[1] : flux[λ]
# get_flux(flux::F, (; λ)::OIDataPoint{U, V, L, Nothing, Nothing}) where {U, V, L, F <: AbstractInterpolation} = flux(λ)
get_flux(flux::Interpolate, (; λ)::OIDataPoint{U, V, L, Nothing, Nothing}) where {U, V, L} = flux(λ)
get_flux(flux::Interpolate, λ::Number) = flux(λ)
