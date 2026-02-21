# reconstruct_hmc.jl
#
# Bayesian image reconstruction via Hamiltonian Monte Carlo.
#
# Algorithm:
#   1. Build an OIPosterior in log-space parameterization.
#   2. Run Pathfinder (multipathfinder) for variational initialization:
#        - obtains a good starting point y_init in log-space
#        - estimates per-parameter variance for the mass matrix
#   3. Set up AdvancedHMC NUTS with the Pathfinder mass matrix and
#      StanHMCAdaptor for joint step-size + mass-matrix adaptation.
#   4. Draw samples and convert back to image space.

"""
    hmc_reconstruct(x_start, data, ft; kwargs...) -> NamedTuple

Run HMC-based posterior sampling for optical interferometric image reconstruction.

# Arguments
- `x_start::Matrix{Float64}` : starting image (will be used if `init_image` not provided)
- `data::OIdata`             : interferometric data
- `ft`                       : NFFT plan (from `setup_nfft`) or DFT matrix

# Keyword arguments
- `nx::Int`                 : image side length (default: `size(x_start, 1)`)
- `weights`                 : `[w_vis, w_v2, w_t3]` chi² weights (default: `[1,1,1]`)
- `regularizers`            : regularizer list (same format as `crit_fg`)
- `vonmises::Bool`          : use von Mises likelihood for closure phases (default: `false`)
- `n_samples::Int`          : number of HMC samples to draw (default: `1000`)
- `n_adapts::Int`           : number of adaptation steps (default: `500`)
- `n_pathfinder_runs::Int`  : number of independent Pathfinder runs (default: `4`)
- `n_pathfinder_draws::Int` : samples from Pathfinder for mass-matrix estimation (default: `200`)
- `δ::Float64`              : target acceptance rate for NUTS (default: `0.8`)
- `init_image`              : optional initial image; if `nothing` uses `x_start`
- `drop_warmup::Bool`       : discard adaptation samples from returned draws (default: `true`)
- `verb::Bool`              : print progress (default: `false`)
- `rng`                     : random number generator (default: `Random.default_rng()`)

# Returns
A `NamedTuple` with fields:
- `mean_image`  : posterior mean image (nx × nx, unit flux)
- `std_image`   : posterior standard deviation per pixel (nx × nx)
- `map_image`   : MAP estimate (unit flux) found during Pathfinder initialization
- `samples`     : matrix of raw log-space samples, size (nx², n_kept)
- `image_samples`: array of image-space samples, size (nx, nx, n_kept)
- `stats`       : AdvancedHMC sampling statistics
- `posterior`   : the `OIPosterior` object
"""
function hmc_reconstruct(
    x_start::Matrix{Float64},
    data::OIdata,
    ft;
    nx::Int                = size(x_start, 1),
    weights                = [1.0, 1.0, 1.0],
    regularizers           = [],
    vonmises::Bool         = false,
    n_samples::Int         = 1000,
    n_adapts::Int          = 500,
    n_pathfinder_runs::Int = 4,
    n_pathfinder_draws::Int = 200,
    δ::Float64             = 0.8,
    init_image             = nothing,
    drop_warmup::Bool      = true,
    verb::Bool             = false,
    rng                    = Random.default_rng(),
)
    # ── 1. Build posterior ────────────────────────────────────────────────────
    posterior = OIPosterior(data, ft, nx;
                            regularizers=regularizers,
                            weights=Float64.(weights),
                            vonmises=vonmises)

    # ── 2. Convert starting image to log-space ────────────────────────────────
    x0 = isnothing(init_image) ? x_start : init_image
    # Add small floor to avoid log(0)
    x0_safe = max.(x0, 1e-30 * maximum(x0))
    y0 = log.(vec(x0_safe))

    # ── 3. Pathfinder initialization ──────────────────────────────────────────
    verb && println("Running multipathfinder ($n_pathfinder_runs runs)...")
    pf_result = multipathfinder(posterior, n_pathfinder_draws;
                                 n_runs=n_pathfinder_runs,
                                 rng=rng)

    # draws is (dim, n_draws); use the first draw as starting point
    y_init = vec(pf_result.draws[:, 1])

    # Diagonal variance of Pathfinder draws → mass matrix
    M_diag = vec(var(pf_result.draws, dims=2))
    M_diag = max.(M_diag, 1e-10)   # numerical floor

    # MAP estimate: take the draw with highest log-posterior
    log_posteriors = [LogDensityProblems.logdensity(posterior, pf_result.draws[:, k])
                      for k in axes(pf_result.draws, 2)]
    best_idx = argmax(log_posteriors)
    y_map = vec(pf_result.draws[:, best_idx])

    # ── 4. AdvancedHMC setup ──────────────────────────────────────────────────
    verb && println("Setting up NUTS sampler...")

    # Scalar wrappers for AdvancedHMC low-level API
    ℓπ(y)  = LogDensityProblems.logdensity(posterior, y)
    ∂ℓπ∂θ(y) = LogDensityProblems.logdensity_and_gradient(posterior, y)

    metric    = DiagEuclideanMetric(M_diag)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)

    # Find a reasonable initial step size
    initial_ε = find_good_stepsize(hamiltonian, y_init)
    integrator = Leapfrog(initial_ε)

    # NUTS kernel with multinomial tree sampling
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

    # Stan-style adaptor: jointly adapts mass matrix and step size
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric),
        StepSizeAdaptor(δ, integrator)
    )

    # ── 5. Sample ─────────────────────────────────────────────────────────────
    verb && println("Running NUTS ($n_adapts adapts + $n_samples samples)...")
    n_total = n_adapts + n_samples

    samples_raw, stats = sample(
        rng, hamiltonian, kernel, y_init, n_total, adaptor, n_adapts;
        drop_warmup=drop_warmup, verbose=verb, progress=verb
    )

    # samples_raw is a vector of vectors; stack into matrix (dim × n_kept)
    samples_mat = reduce(hcat, samples_raw)   # (nx², n_kept)

    # ── 6. Convert to image space ─────────────────────────────────────────────
    n_kept = size(samples_mat, 2)
    image_samples = zeros(nx, nx, n_kept)
    for k in 1:n_kept
        x_k = reshape(exp.(samples_mat[:, k]), nx, nx)
        image_samples[:, :, k] = x_k ./ sum(x_k)
    end

    mean_image = mean(image_samples, dims=3)[:, :, 1]
    std_image  = std(image_samples, dims=3)[:, :, 1]

    # MAP image from Pathfinder (unit flux)
    x_map_unnorm = reshape(exp.(y_map), nx, nx)
    map_image    = x_map_unnorm ./ sum(x_map_unnorm)

    return (
        mean_image   = mean_image,
        std_image    = std_image,
        map_image    = map_image,
        samples      = samples_mat,
        image_samples = image_samples,
        stats        = stats,
        posterior    = posterior,
    )
end
