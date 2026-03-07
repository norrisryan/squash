# reconstruct_hmc.jl
#
# Bayesian image reconstruction via Hamiltonian Monte Carlo.
#
# Algorithm:
#   1. Build an OIPosterior in log-space parameterization.
#   2. For each chain: run SA (sa_init) to find the mode basin, then MAP polish.
#      Each chain gets its own ft deepcopy and RNG, so independent SA trajectories
#      may find different modes — correct behavior for multimodal posteriors.
#   3. Run n_chains independent NUTS chains in parallel (Threads.@threads), each
#      with its own ft copy, OIPosterior, diagonal mass matrix, and adaptor.
#      The StanHMCAdaptor tunes mass matrix and step size during warmup.
#   4. Filter failed chains: acceptance rate < 0.3 or > 5% divergent transitions.
#   5. Compute R-hat (Gelman-Rubin) and ESS diagnostics across converged chains.
#   6. Return combined samples + diagnostics NamedTuple.

"""
    initialize_chain(data, ft, nx, regularizers, weights, rng;
                     n_sa_steps, T_start, n_map_iter, prior_image,
                     use_centroid_correction, verb)

Initialize a single HMC chain via SA + MAP polish (two passes with optional centroid
correction between them).

Phase 1: `sa_init` to find the mode basin (stochastic; each chain's rng gives a
different SA trajectory and may find a different mode).
Phase 2: MAP polish pass 1 from the SA result (and optionally from `prior_image`).
Phase 3 (when `use_centroid_correction=true`): shift the flux-weighted centroid to
the field centre via `circshift`, then MAP polish pass 2 from the corrected image.

**On centroid correction**: V2 and T3phi observables are exactly translation-invariant —
any translated copy of the true image has identical chi2. The centering regularizer alone
cannot reliably break this degeneracy during gradient optimization because the OITOOLS
centering gradient formula is incompatible with log-space sign-gradient SA: it pushes the
wrong pixels in log-space and does not effectively relocate flux between distant positions.
The `circshift` correction directly moves flux to the centred position and is robust for
compact sources (point sources, binaries, stellar surfaces).

Set `use_centroid_correction=false` for extended sources with off-centre morphology where
the flux-weighted centroid does not coincide with the true source centre.

Returns `(y_init, chi2)` — log-space vector and chi2 of the initialization.
"""
function initialize_chain(data, ft, nx, regularizers, weights, rng;
                          n_sa_steps::Int           = 3000,
                          T_start::Float64           = 50.0,
                          n_map_iter::Int            = 200,
                          prior_image                = nothing,
                          use_centroid_correction::Bool = true,
                          verb::Bool                 = false)
    w = Float64.(weights)

    # Phase 1: SA to find the mode basin
    x_sa = sa_init(data, ft, nx;
                   regularizers = regularizers,
                   weights      = w,
                   n_steps      = n_sa_steps,
                   T_start      = T_start,
                   rng          = rng,
                   verb         = verb)

    # Phase 2: MAP polish pass 1 — best of SA start and optional prior_image start
    x_sa_candidates = [x_sa]
    if !isnothing(prior_image)
        x_prior = max.(prior_image, 1e-30); x_prior ./= sum(x_prior)
        push!(x_sa_candidates, x_prior)
    end

    x_map1_best    = x_sa_candidates[1]
    chi2_map1_best = Inf
    g_tmp          = zeros(nx, nx)
    for x_cand in x_sa_candidates
        x_m = map_optimize(x_cand, data, ft;
                           regularizers = regularizers,
                           weights      = w,
                           maxiter      = n_map_iter)
        fill!(g_tmp, 0.0)
        c2 = chi2_fg(x_m, g_tmp, ft, data; weights=w, verb=false)
        if c2 < chi2_map1_best
            chi2_map1_best = c2
            x_map1_best    = x_m
        end
    end

    if use_centroid_correction
        # Phase 3: centroid correction — shift flux-weighted centroid to field centre,
        # then MAP polish pass 2.
        # Necessary because V2+T3phi are translation-invariant: SA finds the correct
        # relative morphology but at an arbitrary translation. circshift moves the entire
        # image so its centroid lands at (nx+1)/2, then MAP pass 2 refines the position.
        # For extended off-centre sources set use_centroid_correction=false.
        cx = sum(i * x_map1_best[i, j] for i in 1:nx, j in 1:nx)
        cy = sum(j * x_map1_best[i, j] for i in 1:nx, j in 1:nx)
        center  = (nx + 1) / 2.0
        shift_r = round(Int, center - cx)
        shift_c = round(Int, center - cy)
        x_corrected = circshift(x_map1_best, (shift_r, shift_c))
        x_corrected = max.(x_corrected, 1e-30); x_corrected ./= sum(x_corrected)

        x_best = map_optimize(x_corrected, data, ft;
                              regularizers = regularizers,
                              weights      = w,
                              maxiter      = n_map_iter)
    else
        x_best = x_map1_best
    end

    x_best = max.(x_best, 1e-30)
    x_best ./= sum(x_best)

    g    = zeros(nx, nx)
    chi2 = chi2_fg(x_best, g, ft, data; weights=w, verb=false)
    verb && println("  Chain init: chi2=$(round(chi2; digits=1))")

    return log.(vec(x_best)), chi2
end

"""
    compute_rhat_ess(chain_mats)

Compute Gelman-Rubin R-hat and effective sample size (ESS) per dimension from a
vector of (dim × n_samples) sample matrices, one per converged chain.

Returns `(rhat, ess)` — both are length-dim vectors.
"""
function compute_rhat_ess(chain_mats)
    m = length(chain_mats)
    n = size(chain_mats[1], 2)

    means_mat = reduce(hcat, [vec(mean(chain_mats[c]; dims=2)) for c in 1:m])
    vars_mat  = reduce(hcat, [vec(var(chain_mats[c];  dims=2)) for c in 1:m])

    psi_bar = vec(mean(means_mat; dims=2))
    B       = (n / (m - 1)) .* vec(sum((means_mat .- psi_bar) .^ 2; dims=2))
    W       = vec(mean(vars_mat; dims=2))
    var_hat = ((n - 1) / n) .* W .+ B ./ n

    rhat = sqrt.(var_hat ./ max.(W, 1e-30))
    ess  = m * n .* W ./ max.(var_hat, 1e-30)

    return rhat, ess
end

"""
    hmc_reconstruct(x_start, data, ft; kwargs...) -> NamedTuple

Run HMC-based posterior sampling for optical interferometric image reconstruction
using multiple independent chains, each initialized by SA + MAP polish.

# Arguments
- `x_start::Matrix{Float64}` : fallback starting image (unused; kept for API compat)
- `data::OIdata`             : interferometric data
- `ft`                       : NFFT plan (from `setup_nfft`) or DFT matrix

# Keyword arguments
- `nx::Int`              : image side length (default: `size(x_start, 1)`)
- `weights`              : `[w_v2, w_t3amp, w_t3phi]` chi² weights (default: `[1,1,1]`)
- `regularizers`         : regularizer list (same format as `crit_fg`)
- `vonmises::Bool`       : use von Mises likelihood for closure phases (default: `false`)
- `n_samples::Int`       : HMC samples per chain (default: `1000`)
- `n_adapts::Int`        : adaptation steps per chain (default: `500`)
- `n_chains::Int`        : number of independent chains (default: `4`)
- `n_sa_steps::Int`      : SA steps for chain initialization (default: `3000`)
- `T_start::Float64`     : SA initial temperature in chi²_red units (default: `50.0`)
- `n_map_iter::Int`      : MAP polish iterations after SA (default: `200`)
- `prior_image`                   : optional prior image for initialization comparison
- `use_centroid_correction::Bool` : shift flux centroid to field centre between MAP passes
                                    (default: `true`; set `false` for off-centre extended sources)
- `δ::Float64`                    : target acceptance rate for NUTS (default: `0.8`)
- `max_tree_depth::Int`  : NUTS maximum tree depth (default: `10`)
- `drop_warmup::Bool`    : discard adaptation samples from returned draws (default: `true`)
- `verb::Bool`           : print progress (default: `false`)
- `rng`                  : random number generator (default: `Random.default_rng()`)

# Returns
A `NamedTuple` with fields:
- `mean_image`   : posterior mean image (nx × nx, unit flux) from converged chains
- `std_image`    : posterior std per pixel (nx × nx) from converged chains
- `map_image`    : best MAP image found during initialization (unit flux)
- `samples`      : combined log-space samples (nx², n_converged × n_samples)
- `image_samples`: combined image-space samples (nx, nx, n_converged × n_samples)
- `stats`        : list of per-chain AdvancedHMC statistics (converged chains only)
- `posterior`    : the `OIPosterior` built from `ft` (shared, for reference)
- `diagnostics`  : NamedTuple with R-hat, ESS, and per-chain acceptance/divergence info
"""
function hmc_reconstruct(
    x_start::Matrix{Float64},
    data::OIdata,
    ft;
    nx::Int               = size(x_start, 1),
    weights               = [1.0, 1.0, 1.0],
    regularizers          = [],
    vonmises::Bool        = false,
    n_samples::Int        = 1000,
    n_adapts::Int         = 500,
    n_chains::Int         = 4,
    n_sa_steps::Int       = 3000,
    T_start::Float64      = 50.0,
    n_map_iter::Int       = 200,
    prior_image                  = nothing,
    use_centroid_correction::Bool = true,
    δ::Float64                   = 0.8,
    max_tree_depth::Int          = 10,
    drop_warmup::Bool            = true,
    verb::Bool                   = false,
    rng                          = Random.default_rng(),
)
    # ── 1. Reference posterior (shared read-only, returned for inspection) ─────
    posterior = OIPosterior(data, ft, nx;
                            regularizers = regularizers,
                            weights      = Float64.(weights),
                            vonmises     = vonmises)

    # ── 2. Per-chain RNGs seeded from main rng ────────────────────────────────
    chain_rngs = [MersenneTwister(rand(rng, UInt32)) for _ in 1:n_chains]

    # ── 3. Pre-allocate per-chain ft copies and result slots ──────────────────
    # ft deepcopies ensure each thread has independent NFFT workspace.
    ft_copies = [deepcopy(ft) for _ in 1:n_chains]

    chain_y_inits     = Vector{Vector{Float64}}(undef, n_chains)
    chain_best_chi2   = fill(Inf, n_chains)
    chain_samples_raw = Vector{Any}(undef, n_chains)
    chain_stats_raw   = Vector{Any}(undef, n_chains)

    verb && println("Running $n_chains chains ($n_adapts adapts + $n_samples samples each)...")
    verb && println("  SA init: n_sa_steps=$n_sa_steps, T_start=$T_start, " *
                    "n_map_iter=$n_map_iter per chain")

    # ── 4. Run chains in parallel ─────────────────────────────────────────────
    Threads.@threads for c in 1:n_chains
        ft_c = ft_copies[c]

        # ── 4a. SA + MAP initialization ───────────────────────────────────────
        y_init_c, chi2_c = initialize_chain(
            data, ft_c, nx, regularizers, weights, chain_rngs[c];
            n_sa_steps               = n_sa_steps,
            T_start                  = T_start,
            n_map_iter               = n_map_iter,
            prior_image              = prior_image,
            use_centroid_correction  = use_centroid_correction,
            verb                     = verb,
        )
        chain_y_inits[c]   = y_init_c
        chain_best_chi2[c] = chi2_c

        # ── 4b. Build per-chain posterior and HMC objects ─────────────────────
        posterior_c = OIPosterior(data, ft_c, nx;
                                  regularizers = regularizers,
                                  weights      = Float64.(weights),
                                  vonmises     = vonmises)

        ℓπ(y)    = LogDensityProblems.logdensity(posterior_c, y)
        ∂ℓπ∂θ(y) = LogDensityProblems.logdensity_and_gradient(posterior_c, y)

        metric_c      = DiagEuclideanMetric(ones(nx^2) ./ nx^2)
        hamiltonian_c = Hamiltonian(metric_c, ℓπ, ∂ℓπ∂θ)

        y_init_c     = chain_y_inits[c]
        initial_ε_c  = find_good_stepsize(hamiltonian_c, y_init_c)
        integrator_c = Leapfrog(initial_ε_c)

        kernel_c = HMCKernel(Trajectory{MultinomialTS}(
            integrator_c, GeneralisedNoUTurn(max_depth = max_tree_depth)
        ))

        adaptor_c = StanHMCAdaptor(
            MassMatrixAdaptor(metric_c),
            StepSizeAdaptor(δ, integrator_c),
        )

        # ── 4c. Draw samples ──────────────────────────────────────────────────
        samples_c, stats_c = sample(
            chain_rngs[c], hamiltonian_c, kernel_c,
            y_init_c, n_adapts + n_samples, adaptor_c, n_adapts;
            drop_warmup = drop_warmup, verbose = false, progress = false,
        )

        chain_samples_raw[c] = samples_c
        chain_stats_raw[c]   = stats_c
    end

    # ── 5. Per-chain diagnostics and failure filtering ─────────────────────────
    chain_accept_rates = zeros(n_chains)
    chain_divergences  = zeros(Int, n_chains)
    converged_mask     = fill(false, n_chains)

    for c in 1:n_chains
        stats_c  = chain_stats_raw[c]
        accept_c = mean(s.acceptance_rate for s in stats_c)
        ndiv_c   = sum(s.numerical_error  for s in stats_c)
        chain_accept_rates[c] = accept_c
        chain_divergences[c]  = ndiv_c
        converged_mask[c]     = accept_c >= 0.3 && ndiv_c <= 0.05 * n_samples
    end

    n_converged = sum(converged_mask)
    n_failed    = n_chains - n_converged

    if n_converged == 0
        @warn "All $n_chains chains failed diagnostics. Using best single chain (highest acceptance)."
        best_c = argmax(chain_accept_rates)
        converged_mask[best_c] = true
        n_converged = 1
        n_failed    = n_chains - 1
    end

    converged_idxs = [c for c in 1:n_chains if converged_mask[c]]

    if verb
        println("Chains converged: $n_converged / $n_chains")
        for c in 1:n_chains
            status = converged_mask[c] ? "OK  " : "FAIL"
            println("  chain $c [$status]: accept=$(round(chain_accept_rates[c]; digits=3))  " *
                    "divergences=$(chain_divergences[c])  " *
                    "init_chi2=$(round(chain_best_chi2[c]; digits=1))")
        end
    end

    # ── 6. Combine samples from converged chains ───────────────────────────────
    chain_mats  = [reduce(hcat, chain_samples_raw[c]) for c in converged_idxs]
    samples_mat = reduce(hcat, chain_mats)

    # ── 7. R-hat and ESS ──────────────────────────────────────────────────────
    if n_converged >= 2
        rhat_vec, ess_vec = compute_rhat_ess(chain_mats)
        rhat_image = reshape(rhat_vec, nx, nx)
        min_ess    = minimum(ess_vec)
    else
        rhat_image = fill(NaN, nx, nx)
        min_ess    = Float64(size(chain_mats[1], 2))
    end

    # ── 8. Convert combined samples to image space ─────────────────────────────
    n_kept = size(samples_mat, 2)
    image_samples = zeros(nx, nx, n_kept)
    for k in 1:n_kept
        y_k = samples_mat[:, k]
        x_k = reshape(exp.(y_k .- maximum(y_k)), nx, nx)
        image_samples[:, :, k] = x_k ./ sum(x_k)
    end

    mean_image = mean(image_samples; dims=3)[:, :, 1]
    std_image  = std(image_samples;  dims=3)[:, :, 1]

    # MAP image: chain with lowest init chi2
    best_map_c   = argmin(chain_best_chi2)
    y_map        = chain_y_inits[best_map_c]
    x_map_unnorm = reshape(exp.(y_map .- maximum(y_map)), nx, nx)
    map_image    = x_map_unnorm ./ sum(x_map_unnorm)

    return (
        mean_image    = mean_image,
        std_image     = std_image,
        map_image     = map_image,
        samples       = samples_mat,
        image_samples = image_samples,
        stats         = [chain_stats_raw[c] for c in converged_idxs],
        posterior     = posterior,
        diagnostics   = (
            rhat               = rhat_image,
            min_ess            = min_ess,
            n_converged        = n_converged,
            n_failed           = n_failed,
            chain_accept_rates = chain_accept_rates,
            chain_divergences  = chain_divergences,
            chain_map_chi2     = chain_best_chi2,
        ),
    )
end

"""
    map_reconstruct(x_start, data, ft; kwargs...) -> Matrix{Float64}

Find the MAP image from a given starting image using L-BFGS in log-space.
For SA-initialized MAP without a starting image, use the `map_reconstruct(data, ft, nx; ...)`
method in reconstruct_map.jl.
"""
function map_reconstruct(
    x_start::Matrix{Float64},
    data::OIdata,
    ft;
    nx::Int        = size(x_start, 1),
    weights        = [1.0, 1.0, 1.0],
    regularizers   = [],
    vonmises::Bool = false,
    maxiter::Int   = 500,
    verb::Bool     = false,
)
    posterior = OIPosterior(data, ft, nx;
                            regularizers = regularizers,
                            weights      = Float64.(weights),
                            vonmises     = vonmises)

    x0_safe = max.(x_start, 1e-30 * max(maximum(x_start), 1.0 / nx^2))
    y0 = log.(vec(x0_safe))

    neg_lp(y) = -LogDensityProblems.logdensity(posterior, y)
    function neg_lp_grad!(g, y)
        _, grad = LogDensityProblems.logdensity_and_gradient(posterior, y)
        g .= -grad
    end

    opt = Optim.optimize(
        neg_lp, neg_lp_grad!, y0,
        Optim.LBFGS(),
        Optim.Options(iterations=maxiter, show_trace=verb),
    )

    verb && println("MAP: converged=$(Optim.converged(opt)), iters=$(Optim.iterations(opt)), f=$(Optim.minimum(opt))")

    y_opt = Optim.minimizer(opt)
    x_map = reshape(exp.(y_opt .- maximum(y_opt)), nx, nx)
    return x_map ./ sum(x_map)
end
