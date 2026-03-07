# reconstruct_map.jl
#
# SA-initialized MAP reconstruction and regularizer comparison utilities.
# Uses extended_crit_fg + OptimPackNextGen.vmlmb so that custom regularizers
# (laplacian, good_roughness, l1l2_wavelet) work in the MAP path.

"""
    extended_crit_fg(x, g, ft, data; weights, regularizers, verb) -> Float64

Combined chi² + regularization criterion for pixel-space MAP optimization.
Handles both OITOOLS built-in and OIImage custom regularizers via
`extended_regularization`. Gradient `g` is filled in-place.

A flux-normalization correction is applied to `g` so that the optimizer
sees a gradient consistent with optimizing over the (automatically
flux-normalized) image simplex.
"""
function extended_crit_fg(x::Matrix{Float64}, g::Matrix{Float64},
                           ft, data::OIdata;
                           weights     = [1.0, 0.0, 1.0],
                           regularizers = [],
                           verb::Bool  = false)
    # chi2 term — fills g in-place
    chi2 = chi2_fg(x, g, ft, data; weights=weights, verb=verb)
    # all regularizers including custom ones — accumulates into g
    reg  = extended_regularization(x, g; regularizers=regularizers, verb=verb)
    # flux normalization correction
    flux = sum(x)
    g .= (g .- sum(x .* g) ./ flux) ./ flux
    return chi2 + reg
end

"""
    map_optimize(x_start, data, ft; regularizers, weights, maxiter, verb) -> Matrix{Float64}

Pixel-space MAP optimization via `OptimPackNextGen.vmlmb`.
Supports all regularizers recognized by `extended_regularization`.
Returns the optimized (unit-flux, non-negative) image.
"""
function map_optimize(x_start::Matrix{Float64}, data, ft;
                      regularizers = [],
                      weights      = [1.0, 0.0, 1.0],
                      maxiter::Int = 500,
                      verb::Bool   = false)
    nx   = size(x_start, 1)
    crit = (x, g) -> extended_crit_fg(reshape(x, nx, nx),
                                       reshape(g, nx, nx),
                                       ft, data;
                                       weights      = weights,
                                       regularizers = regularizers,
                                       verb         = verb)
    x_sol = OptimPackNextGen.vmlmb(crit, vec(x_start);
                                   verb    = verb,
                                   lower   = 0,
                                   maxiter = maxiter,
                                   blmvm   = false)
    x_out = reshape(x_sol, nx, nx)
    x_out = max.(x_out, 0.0)
    s     = sum(x_out)
    return s > 0 ? x_out ./ s : fill(1.0 / nx^2, nx, nx)
end

"""
    map_reconstruct(data, ft, nx; kwargs...) -> NamedTuple

Find the MAP image using SA initialization + single-pass L-BFGS polish.
Supports all regularizers (OITOOLS built-ins and OIImage custom ones).

!!! note "Absolute source positions"
    `map_reconstruct` does **not** apply centroid correction, so the recovered image
    may be translated relative to the true source position (V2 and T3phi are exactly
    translation-invariant). This makes `map_reconstruct` well-suited for:
    - Chi² minimization and regularizer comparison across many parameter sets
    - Extended sources where the flux centroid does not coincide with the source centre
    - Cases where relative morphology matters more than absolute position

    For **absolute source position recovery** (point sources, binaries, stellar surfaces)
    use `hmc_reconstruct`, which applies a `circshift` centroid correction in
    `initialize_chain` to break the translation degeneracy before HMC sampling.

# Keyword arguments
- `weights`         : `[w_v2, w_t3amp, w_t3phi]` chi² weights (default: `[1,1,1]`)
- `regularizers`    : regularizer list (same format as `extended_regularization`)
- `n_sa_steps::Int` : SA steps for initialization (default: `3000`)
- `T_start::Float64`: SA initial temperature in chi²_red units (default: `50.0`)
- `n_map_iter::Int` : MAP L-BFGS iterations (default: `500`)
- `verb::Bool`      : print progress (default: `false`)
- `rng`             : random number generator (default: `Random.default_rng()`)

# Returns
A `NamedTuple` with fields:
- `image`       : MAP image (nx × nx, unit flux)
- `chi2`        : chi² of the MAP image
- `chi2_reduced`: chi² / n_data
- `init_image`  : SA initialization image (before MAP polish)
"""
function map_reconstruct(
    data::OIdata,
    ft,
    nx::Int;
    weights      = [1.0, 1.0, 1.0],
    regularizers = [],
    n_sa_steps::Int  = 3000,
    T_start::Float64 = 50.0,
    n_map_iter::Int  = 500,
    verb::Bool       = false,
    rng              = Random.default_rng(),
)
    w      = Float64.(weights)
    n_data = data.nv2 + data.nt3amp + data.nt3phi

    # Phase 1: SA initialization
    verb && println("  SA init ($n_sa_steps steps, T_start=$T_start)...")
    x_sa = sa_init(data, ft, nx;
                   regularizers = regularizers,
                   weights      = w,
                   n_steps      = n_sa_steps,
                   T_start      = T_start,
                   rng          = rng,
                   verb         = verb)

    init_image = copy(x_sa)

    # Phase 2: MAP polish from SA result
    verb && println("  MAP polish ($n_map_iter iters)...")
    x_map = map_optimize(x_sa, data, ft;
                         regularizers = regularizers,
                         weights      = w,
                         maxiter      = n_map_iter,
                         verb         = false)

    g    = zeros(nx, nx)
    chi2 = chi2_fg(x_map, g, ft, data; weights=w, verb=false)
    chi2_red = n_data > 0 ? chi2 / n_data : NaN

    verb && println("  MAP chi2=$(round(chi2; digits=1))  chi2_red=$(round(chi2_red; digits=3))")

    return (
        image        = x_map,
        chi2         = chi2,
        chi2_reduced = chi2_red,
        init_image   = init_image,
    )
end

"""
    compare_regularizers(data, ft, nx, regularizer_sets; kwargs...) -> Vector{NamedTuple}

Run `map_reconstruct` independently for each regularizer set and return results sorted by chi².

Each element of `regularizer_sets` is a vector of regularizer specs accepted by
`extended_regularization` (supports both OITOOLS built-ins and custom OIImage regularizers).

# Keyword arguments
Same as `map_reconstruct` (except `regularizers`), applied to every run.

# Returns
Vector of `NamedTuple`s (same fields as `map_reconstruct`), one per regularizer set,
sorted by ascending `chi2`. Each result also includes:
- `regularizers` : the regularizer set used
- `rank`         : rank by chi² (1 = best)
"""
function compare_regularizers(
    data::OIdata,
    ft,
    nx::Int,
    regularizer_sets::Vector;
    weights      = [1.0, 1.0, 1.0],
    n_sa_steps::Int  = 3000,
    T_start::Float64 = 50.0,
    n_map_iter::Int  = 500,
    verb::Bool       = false,
    rng              = Random.default_rng(),
)
    n_sets  = length(regularizer_sets)
    results = Vector{Any}(undef, n_sets)

    for k in 1:n_sets
        regs = regularizer_sets[k]
        verb && println("── Regularizer set $k / $n_sets ──────────────────────")
        verb && println("  $regs")

        rng_k = MersenneTwister(rand(rng, UInt32))
        ft_k  = deepcopy(ft)

        r = map_reconstruct(data, ft_k, nx;
                            weights      = weights,
                            regularizers = regs,
                            n_sa_steps   = n_sa_steps,
                            T_start      = T_start,
                            n_map_iter   = n_map_iter,
                            verb         = verb,
                            rng          = rng_k)

        results[k] = merge(r, (regularizers=regs, rank=0))
    end

    order  = sortperm([r.chi2 for r in results])
    sorted = [(merge(results[order[i]], (rank=i,))) for i in 1:n_sets]

    if verb
        println("\n── compare_regularizers summary ────────────────────────")
        println("  Rank | chi2       | chi2_red  | regularizers")
        println("  -----|------------|-----------|-------------")
        for r in sorted
            println("   $(r.rank)   | $(round(r.chi2; digits=1)) | " *
                    "$(round(r.chi2_reduced; digits=3)) | $(r.regularizers)")
        end
    end

    return sorted
end
