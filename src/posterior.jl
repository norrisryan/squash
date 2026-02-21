# posterior.jl
#
# OIPosterior: a LogDensityProblems-compatible struct representing the posterior
# log p(y | data) ∝ exp(-½ χ² - λ·reg) in log-space parameterization.
#
# Parameterization:
#   y ∈ ℝ^(nx²)          log-space parameters (unconstrained)
#   x_unnorm = exp.(y)   unnormalized image (always positive)
#   x = x_unnorm / sum(x_unnorm)  unit-flux normalized image
#
# Log-posterior:
#   log p(y) = -(χ²(x) + reg(x))  + const
#
# Gradient derivation:
#   chi2_fg fills g with ∂χ²/∂z where z = x (normalized) — OITOOLS convention.
#   extended_regularization fills g with ∂reg/∂x (unnormalized).
#   After combining, apply the normalized-image flux correction (same as crit_fg):
#       g_x[i] = (g[i] - sum(x.*g)/flux) / flux   (= ∂(χ²+reg)/∂x_unnorm[i])
#   Chain rule through exp:
#       g_y[i] = g_x[i] * x_unnorm[i]             (= ∂(χ²+reg)/∂y[i])
#   Negate for log-posterior gradient:
#       ∂log p/∂y[i] = -g_y[i]

"""
    OIPosterior

Holds all fixed data needed to evaluate the log-posterior and its gradient for
optical interferometric image reconstruction using HMC.

Fields
------
- `data`        : `OIdata` struct with the interferometric observables
- `ft`          : NFFT plan array returned by `setup_nfft` (or DFT matrix from `setup_dft`)
- `nx`          : image side length (image is `nx × nx`)
- `regularizers`: regularizer specification list, same format as OITOOLS `crit_fg`
- `weights`     : length-3 vector `[w_vis, w_v2, w_t3]` scaling each χ² term
- `vonmises`    : if `true`, use von Mises distribution for closure phases
"""
struct OIPosterior
    data::OIdata
    ft                        # NFFTPlan array or DFT matrix
    nx::Int
    regularizers
    weights::Vector{Float64}
    vonmises::Bool
end

"""
    OIPosterior(data, ft, nx; regularizers=[], weights=[1.0,1.0,1.0], vonmises=false)

Convenience constructor.
"""
function OIPosterior(data::OIdata, ft, nx::Int;
                     regularizers=[], weights=[1.0, 1.0, 1.0], vonmises::Bool=false)
    return OIPosterior(data, ft, nx, regularizers, Float64.(weights), vonmises)
end

# ── LogDensityProblems interface ──────────────────────────────────────────────

LogDensityProblems.dimension(p::OIPosterior) = p.nx^2

LogDensityProblems.capabilities(::Type{OIPosterior}) =
    LogDensityProblems.LogDensityOrder{1}()

"""
    LogDensityProblems.logdensity(p::OIPosterior, y) -> Float64

Evaluate log p(y) = -(χ² + reg) without computing the gradient.
"""
function LogDensityProblems.logdensity(p::OIPosterior, y::AbstractVector{<:Real})
    x = reshape(exp.(y), p.nx, p.nx)
    g = zeros(p.nx, p.nx)
    chi2 = chi2_fg(x, g, p.ft, p.data;
                   weights=p.weights, verb=false, vonmises=p.vonmises)
    fill!(g, 0.0)
    reg  = extended_regularization(x, g; regularizers=p.regularizers, verb=false)
    return -(chi2 + reg)
end

"""
    LogDensityProblems.logdensity_and_gradient(p::OIPosterior, y) -> (lp, g_y)

Evaluate log p(y) and its gradient ∂log p/∂y simultaneously.

Returns `(log_posterior, gradient_wrt_y)` where both are `Float64` / `Vector{Float64}`.
"""
function LogDensityProblems.logdensity_and_gradient(p::OIPosterior,
                                                     y::AbstractVector{<:Real})
    x_unnorm = reshape(exp.(y), p.nx, p.nx)
    flux = sum(x_unnorm)
    x = x_unnorm ./ flux          # normalized image passed to chi2_fg

    g = zeros(p.nx, p.nx)

    # χ² term: g ← ∂χ²/∂z (z = normalized image)
    chi2 = chi2_fg(x, g, p.ft, p.data;
                   weights=p.weights, verb=false, vonmises=p.vonmises)

    # Regularization term: g += ∂reg/∂x  (same convention as crit_fg)
    reg = extended_regularization(x, g; regularizers=p.regularizers, verb=false)

    # Flux-normalization correction (converts combined gradient to ∂/∂x_unnorm)
    # Matches crit_fg:  g ← (g - sum(x.*g)/flux) / flux
    g .= (g .- sum(x_unnorm .* g) ./ flux) ./ flux

    # Chain rule through exp:  ∂/∂y_i = ∂/∂x_unnorm_i * x_unnorm_i
    # Negate because we want gradient of the log-posterior (not the cost)
    g_y = -vec(g) .* vec(x_unnorm)

    return -(chi2 + reg), g_y
end
