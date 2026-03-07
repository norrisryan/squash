# new_regularizers.jl
#
# Additional regularizers to complement those in OITOOLS.
# All follow the OITOOLS convention:
#   - function f(x::Matrix, g::Matrix; kwargs...) -> Float64
#   - x is the (unnormalized) image, g is modified in-place (gradient accumulated)
#   - returns the regularization penalty value

# Names recognized by extended_regularization (not in OITOOLS)
const NEW_REGULARIZER_NAMES = ["laplacian", "good_roughness", "l1l2_wavelet"]

# Names recognized by OITOOLS regularization()
const OITOOLS_REGULARIZER_NAMES = [
    "centering", "tv", "tvsq", "EPLL", "l1l2", "l1l2w",
    "l1hyp", "l2sq", "compactness", "radialvar", "entropy", "support"
]

"""
    laplacian(x, g; verb=false) -> Float64

Bilaplacian roughness penalty: f = sum((Δx)²), gradient = 2·Δ(Δx).

`x` is the image (2-D matrix), `g` is accumulated in-place with the gradient.
"""
function laplacian(x::Matrix{Float64}, g::Matrix{Float64}; verb::Bool=false)
    # Laplacian with periodic boundary (circshift)
    lap = circshift(x, (-1, 0)) .+ circshift(x, (1, 0)) .+
          circshift(x, (0, -1)) .+ circshift(x, (0, 1)) .- 4.0 .* x

    f = sum(lap .^ 2)

    # Gradient of f w.r.t. x is 2 * bilaplacian(x) = 2 * Δ(lap)
    bilap = circshift(lap, (-1, 0)) .+ circshift(lap, (1, 0)) .+
            circshift(lap, (0, -1)) .+ circshift(lap, (0, 1)) .- 4.0 .* lap
    g .+= 2.0 .* bilap

    verb && println("laplacian: ", f)
    return f
end

"""
    good_roughness(x, g; verb=false, ϵ=1e-8) -> Float64

Good's roughness penalty: f = Σᵢⱼ (xᵢ - xⱼ)² / (xᵢ + xⱼ + ϵ)
summed over horizontally and vertically adjacent pixel pairs.

Gradient computed analytically.
"""
function good_roughness(x::Matrix{Float64}, g::Matrix{Float64};
                        verb::Bool=false, ϵ::Float64=1e-8)
    f = 0.0
    nx, ny = size(x)

    # Horizontal neighbours: (i,j) -- (i,j+1)
    xr = circshift(x, (0, -1))   # x shifted right (j → j+1)
    d = x .- xr
    s = x .+ xr .+ ϵ
    f += sum(d .^ 2 ./ s)
    # df/dx_ij  (two contributions: as left pixel and as right pixel of a pair)
    g .+= (2.0 .* d ./ s .- d .^ 2 ./ s .^ 2)    # as left pixel
    g .-= circshift(2.0 .* d ./ s .+ d .^ 2 ./ s .^ 2, (0, 1))  # as right pixel (shift back)

    # Vertical neighbours: (i,j) -- (i+1,j)
    xd = circshift(x, (-1, 0))   # x shifted down (i → i+1)
    d = x .- xd
    s = x .+ xd .+ ϵ
    f += sum(d .^ 2 ./ s)
    g .+= (2.0 .* d ./ s .- d .^ 2 ./ s .^ 2)
    g .-= circshift(2.0 .* d ./ s .+ d .^ 2 ./ s .^ 2, (1, 0))

    verb && println("good_roughness: ", f)
    return f
end

"""
    l1l2_wavelet(x, g; verb=false, ϵ=1e-8, nscales=3) -> Float64

Multiscale l1l2 roughness penalty. For each dyadic scale s = 1, 2, 4, ..., 2^(nscales-1),
compute the l1l2 norm of differences between pixels separated by stride s in both
horizontal and vertical directions.

f = Σ_s Σᵢⱼ sqrt((xᵢⱼ - xᵢ₊ₛ,ⱼ)² + ϵ) + sqrt((xᵢⱼ - xᵢ,ⱼ₊ₛ)² + ϵ) - 2√ϵ

Gradient is accumulated into g in-place.
"""
function l1l2_wavelet(x::Matrix{Float64}, g::Matrix{Float64};
                      verb::Bool=false, ϵ::Float64=1e-8, nscales::Int=3)
    f = 0.0
    for s in 0:(nscales - 1)
        stride = 2^s

        # Horizontal differences at scale stride
        xr = circshift(x, (0, -stride))
        d_h = x .- xr
        r_h = sqrt.(d_h .^ 2 .+ ϵ)
        f += sum(r_h) - length(r_h) * sqrt(ϵ)
        gh = d_h ./ r_h
        g .+= gh
        g .-= circshift(gh, (0, stride))

        # Vertical differences at scale stride
        xd = circshift(x, (-stride, 0))
        d_v = x .- xd
        r_v = sqrt.(d_v .^ 2 .+ ϵ)
        f += sum(r_v) - length(r_v) * sqrt(ϵ)
        gv = d_v ./ r_v
        g .+= gv
        g .-= circshift(gv, (stride, 0))
    end

    verb && println("l1l2_wavelet: ", f)
    return f
end

"""
    extended_regularization(x, reg_g; regularizers=[], verb=true) -> Float64

Extended regularization dispatcher that handles both OITOOLS built-in regularizers
and the new ones defined in this file (laplacian, good_roughness, l1l2_wavelet).

Each entry in `regularizers` is a vector/tuple whose first element is the regularizer
name (String) and second element is the weight (Float64), followed by any additional
parameters specific to that regularizer (matching OITOOLS conventions).

`reg_g` is accumulated in-place with the gradient.
"""
function extended_regularization(x::Matrix{Float64}, reg_g::Matrix{Float64};
                                  regularizers=[], verb::Bool=true)
    isempty(regularizers) && return 0.0

    f = 0.0

    # Partition into OITOOLS and new regularizers
    oitools_regs = filter(r -> r[1] in OITOOLS_REGULARIZER_NAMES, regularizers)
    new_regs = filter(r -> r[1] in NEW_REGULARIZER_NAMES, regularizers)
    unknown = filter(r -> !(r[1] in OITOOLS_REGULARIZER_NAMES) &&
                          !(r[1] in NEW_REGULARIZER_NAMES), regularizers)

    if !isempty(unknown)
        names = join([r[1] for r in unknown], ", ")
        error("Unknown regularizer(s): $names")
    end

    # Delegate OITOOLS regularizers to the OITOOLS function
    if !isempty(oitools_regs)
        f += regularization(x, reg_g; regularizers=oitools_regs, verb=verb)
    end

    # Handle new regularizers
    for ireg in new_regs
        rname = ireg[1]
        rweight = length(ireg) >= 2 ? Float64(ireg[2]) : 1.0
        g_tmp = zeros(size(x))

        if rname == "laplacian"
            f += rweight * laplacian(x, g_tmp; verb=verb)
        elseif rname == "good_roughness"
            ε = length(ireg) >= 3 ? Float64(ireg[3]) : 1e-8
            f += rweight * good_roughness(x, g_tmp; verb=verb, ϵ=ε)
        elseif rname == "l1l2_wavelet"
            ε = length(ireg) >= 3 ? Float64(ireg[3]) : 1e-8
            ns = length(ireg) >= 4 ? Int(ireg[4]) : 3
            f += rweight * l1l2_wavelet(x, g_tmp; verb=verb, ϵ=ε, nscales=ns)
        end

        reg_g .+= rweight .* g_tmp
    end

    return f
end
