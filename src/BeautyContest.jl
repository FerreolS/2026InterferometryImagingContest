module BeautyContest


using StellarTracks, BolometricCorrections, InitialMassFunctions
using TypedTables: Table
using Unitful, UnitfulAstro, UnitfulAngles
using Korg
using LinearInterpolations
using AstroFITS, OIFITS
using Parameters
using AxisArrays
using Serialization
using BlockArrays
using LinearAlgebra
using StatsBase

import UnitfulAngles: mas

using ConcreteStructs
using Interpolations
using LinearInterpolations
using Optimisers
using Unitful

export GaussianFunction, get_vis, make_V2, visgen, SNR_V2, make_t3, Instrument, make_vis,
    fluxtoJansky, get_photonflux, fluxtophoton, photontoJansky, make_t3phi
export makeV2corr_Kammerer2020, makeT3corr_Kammerer2020, sparsify_correlation_matrix
export OIDataPoint, Punct, get_flux

include("models.jl")

@with_kw struct Instrument

    R = 500                 #spectral resolution

    fov = 100.0u"mas"        # field of view
    fiber_fwhm = 65.0u"mas" # fiber lobe fwhm (or 90?)

    Atel = 49.29u"m^2"        # collecting Area
    Ttel = 0.28             # transmission through the optical train
    Tfiber = 0.8            # transmission of the fiber_fwhm
    #Tgrism = 0.5			# grism transmission
    Tgrism = Serialization.deserialize("data/throughput_post-intervention-MED")
    Δλ = 2.18884049845577u"nm" # spectral bin width

    SR = 0.67                # strehl Ratio
    DIT = 100u"s"            # integration time
    NDIT = 4                 # number of integration


    QE = 0.81                 # detector quantum efficiency


    bkg_param = Serialization.deserialize("data/background_noise_post-intervention_2021")["MEDIUM"]["COMBINED"]
    #σRN = 3.5                   # e- / pixel Read out noise
    σRN = bkg_param[:, 3] ./ ustrip(u"s", DIT) .^ 0.5 .+ bkg_param[:, 4]
    nprof = 4 * sum((GaussianFunction(1.7, -1.5:1.0:1.5) ./ sum(GaussianFunction(1.7, -1.5:1.0:1.5))) .^ 2)

    σch = sqrt(nprof) .* σRN # Read out noise per channel

    Ksky = 14.9u"arcsecond^-2" # sky brightness in K band

end

function GaussianFunction(fwhm, position)
    σ = fwhm / (2 * sqrt(2 * log(2)))
    r² = abs2.(position)
    return exp.(-r² ./ (2 * σ^2))
end

function get_vis(obj, ucoord, vcoord, λ)
    d = OIDataPoint(u = ucoord ./ λ |> u"rad^-1", v = vcoord ./ λ |> u"rad^-1", λ = λ)
    return obj(d) ./ get_flux(obj, d)
end

function make_vis(obj, ucoord, vcoord, tλ)
    V = zeros(ComplexF64, length(tλ), length(ucoord))
    for (idxu, (u, v)) in enumerate(zip(ucoord, vcoord))
        for (idxλ, λ) in enumerate(tλ)
            V[idxλ, idxu] = get_vis(obj, u, v, λ)
        end
    end
    return V
end

function visgen(stars, λ, urange)
    visarray = Matrix{ComplexF64}(undef, length(urange), length(urange))
    @inbounds for (i, u) in enumerate(urange)
        for (j, v) in enumerate(urange)
            visarray[i, j] = get_vis(stars, u, v, λ)
        end
    end
    return visarray
end

function make_V2(obj, ucoord, vcoord, tλ; Varch = nothing, flux = 1, nsample = 1_000)
    V = zeros(ComplexF64, length(tλ), length(ucoord))

    for (idxu, (u, v)) in enumerate(zip(ucoord, vcoord))
        for (idxλ, λ) in enumerate(tλ)
            V[idxλ, idxu] = get_vis(obj, u, v, λ)
        end
    end
    if isnothing(Varch)
        return abs2.(V)
    end

    corrV = flux .* V
    σv = sqrt.(1 / 2 .* Varch)
    vr = real.(corrV) .+ σv .* randn(Float64, length(tλ), length(ucoord), nsample)
    vi = imag.(corrV) .+ σv .* randn(Float64, length(tλ), length(ucoord), nsample)
    t = 1 / 2 .* flux  # /3?
    σt = sqrt.(18 .* Varch) ./ 12
    t1 = t .+ σt .* randn(Float64, length(tλ), length(ucoord), nsample)
    t2 = t .+ σt .* randn(Float64, length(tλ), length(ucoord), nsample)
    v2 = (vr .^ 2 .+ vi .^ 2) ./ (4 .* t1 .* t2)
    v2err = dropdims(std(v2; dims = 3), dims = 3)
    return abs2.(V), v2err
end


function SNR_V2(V2, N, N_bkg, NDIT, σ_RN)
    """
    Equation 19: SNR for squared visibility

    Parameters:
    - V2: squared visibility (0 to 1)
    - N: total photon count (signal)
    - N_bkg: background photon count
    - NDIT: number of integrations
    - σ_RN: read noise in electrons
    """
    numerator = sqrt.(NDIT) .* N^2 .* V2

    nn = N .+ N_bkg

    # term1 = 2 * (N + N_bkg)^3 * V2

    # term2 = (N + N_bkg)^2
    # term3 = σ_RN^2 * (2 * (N + N_bkg)^2 * V2 + 2 * (N + N_bkg) + 1/4)
    # term4 = σ_RN^4

    #denominator = sqrt(term1 + term2 + term3 + term4)
    denominator = sqrt.(2 .* nn .^ 3 .* V2 .+ nn .^ 2 .+ σ_RN .^ 2 .* (2 .* nn .^ 2 .* V2 .+ 2 .* nn .+ 1 ./ 4) .+ σ_RN .^ 4)
    return numerator ./ denominator
end

function make_t3(obj, u1coord::Number, v1coord::Number, u2coord::Number, v2coord::Number, λ::Number)
    v1 = get_vis(obj, u1coord, v1coord, λ)
    v2 = get_vis(obj, u2coord, v2coord, λ)
    v3 = get_vis(obj, u1coord .+ u2coord, v1coord .+ v1coord, λ)
    t3 = v1 .* v2 .* conj.(v3)
    return t3
end


function make_t3phi(obj, u1coord, v1coord, u2coord, v2coord, tλ; Varch = nothing, flux, nsample = 1_000)

    v1 = zeros(ComplexF64, length(tλ), length(u1coord))
    v2 = zeros(ComplexF64, length(tλ), length(u1coord))
    v3 = zeros(ComplexF64, length(tλ), length(u1coord))
    for (idxu, (u, v)) in enumerate(zip(u1coord, v1coord))
        for (idxλ, λ) in enumerate(tλ)
            v1[idxλ, idxu] = get_vis(obj, u, v, λ)
        end
    end

    for (idxu, (u, v)) in enumerate(zip(u2coord, v2coord))
        for (idxλ, λ) in enumerate(tλ)
            v2[idxλ, idxu] = get_vis(obj, u, v, λ)
        end
    end


    for (idxu, (u, v)) in enumerate(zip(u1coord .+ u2coord, v1coord .+ v2coord))
        for (idxλ, λ) in enumerate(tλ)
            v3[idxλ, idxu] = get_vis(obj, u, v, λ)
        end
    end
    t3 = angle.(v1 .* v2 .* conj.(v3))
    if isnothing(Varch)
        return t3 .|> u"deg"
    end

    v1 = flux .* v1
    v2 = flux .* v2
    v3 = flux .* v3

    σ = sqrt.(2 .* Varch)
    b1 = v1 .+ σ .* randn(ComplexF64, length(tλ), length(u1coord), nsample)
    b2 = v2 .+ σ .* randn(ComplexF64, length(tλ), length(u1coord), nsample)
    b3 = v3 .+ σ .* randn(ComplexF64, length(tλ), length(u1coord), nsample)
    c = angle.(b1 .* b2 .* conj.(b3) .* cis.(-t3))
    t3phierr = dropdims(std(c; dims = 3), ; dims = 3)
    return t3 .|> u"deg", t3phierr .|> u"deg"
end


function get_photonflux(Kmag, oiflux)
    Krange = 13:190 # for 390nm in MED resolution
    return upreferred(45.6873034694u"1/cm^2/s/angstrom" * 10^(-0.4 * Kmag)) .* oiflux ./ mean(oiflux[Krange]) #ou 47.4099226559661
end


function fluxtoJansky(photonflux, λ)
    # Constants
    h = 6.62607015e-34u"J*s"
    c = 2.99792458e8u"m*s^-1"
    return photonflux .* h .* λ .|> u"Jy"
end

function fluxtophoton(photonflux, inst::Instrument, Tsky)
    @unpack_Instrument inst
    #return upreferred.(Atel * DIT * Δλ .* photonflux)

    return upreferred.(QE * Ttel * Tfiber * Atel * DIT * Δλ .* Tgrism .* Tsky .* photonflux)

end

function photontoJansky(photons, inst, λ, Tsky)
    @unpack_Instrument inst
    flux = photons ./ upreferred.(QE * Ttel * Tfiber * Atel * DIT * Δλ .* Tgrism .* Tsky)
    return fluxtoJansky(flux, λ)
end

function makeV2corr_Kammerer2020(b, Λ, x::T) where {T <: AbstractFloat}
    X1 = x .* ones(T, Λ, Λ)
    diagview(X1) .= T(1)
    X2 = x / 2 .* ones(T, Λ, Λ)
    Z = zeros(T, Λ, Λ)
    A = [Z, X2, X1]
    return LinearAlgebra.Symmetric(collect(mortar([A[1 + length(filter(x -> x ∈ b[:, i], b[:, j]))] for i in Base.axes(b, 2) , j in Base.axes(b, 2)])))
end

function makeT3corr_Kammerer2020(B, Λ, y::T) where {T <: AbstractFloat}
    Y1 = y .* ones(T, Λ, Λ)
    diagview(Y1) .= T(1)
    Y2 = y / 3 .* ones(T, Λ, Λ)
    diagview(Y2) .= T(1 / 3)
    Y3 = -Y2
    A = [Y1, Y2, Y3]
    M = [
        1 2 3 2
        2 1 2 3
        3 2 1 2
        2 3 2 1
    ]
    return LinearAlgebra.Symmetric(collect(mortar(A[M])))
end

function sparsify_correlation_matrix(corr)
    M, N = size(corr)
    M == N || error("M ≠ N)")
    maxel = round(Int, (5 * (5 - 1)) / 2, RoundUp)
    IINDX = Vector{Int}()
    sizehint!(IINDX, maxel)
    JINDX = Vector{Int}()
    sizehint!(JINDX, maxel)
    CORR = Vector{Float64}()
    sizehint!(CORR, maxel)
    for m in 1:M
        for n in (m + 1):N
            if corr[m, n] ≠ 0
                push!(IINDX, m)
                push!(JINDX, n)
                push!(CORR, corr[m, n])
            end
        end
    end
    return IINDX, JINDX, CORR
end

end
