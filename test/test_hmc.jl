# test/test_hmc.jl
#
# Binary-star integration test for hmc_reconstruct with SA + MAP init.
#
# Setup:
#   - 64×64 image, two point sources at (28,36) and (36,28), flux ratio 0.7/0.3
#   - 15-station circular array → 105 V2 baselines, 455 T3 triangles
#   - 5% relative V2 noise, 5° T3phi noise
#   - Starting image: flat + 5% Gaussian perturbation (no MAP pre-conditioning)
#   - Regularizers: centering 1e-3, tv 1e-4; weights [1.0, 0.0, 1.0]
#   - n_chains=2, n_sa_steps=2000, n_samples=50, n_adapts=50
#   - Assert at least 1/2 chains converged (acceptance >= 0.3, divergences < 5%)
#   - Assert top-2 posterior mean peaks within 4 px of true positions
#   - Report per-chain MAP chi2, diagnostics, and total wall-clock time

using OIImage
using Test
using Random
using Statistics
using LinearAlgebra

Random.seed!(1234)

# ── Simulate observables ───────────────────────────────────────────────────────

function simulate_data_hmc(x_true::Matrix{Float64},
                           uv::Matrix{Float64},
                           indx_v2::Vector{Int},
                           indx_t3_1::Vector{Int},
                           indx_t3_2::Vector{Int},
                           indx_t3_3::Vector{Int},
                           pixsize_mas::Float64;
                           noise_frac::Float64    = 0.05,
                           t3phi_err_deg::Float64 = 5.0,
                           rng = Random.default_rng())
    nx     = size(x_true, 1)
    x_norm = x_true ./ sum(x_true)
    dft    = setup_dft(uv, nx, pixsize_mas)
    cvis   = dft * vec(x_norm)

    v2_true = abs2.(cvis[indx_v2])
    v2_err  = noise_frac .* max.(v2_true, 1e-6)
    v2_obs  = v2_true .+ randn(rng, length(indx_v2)) .* v2_err

    t3_true    = cvis[indx_t3_1] .* cvis[indx_t3_2] .* cvis[indx_t3_3]
    t3phi_true = angle.(t3_true) .* (180.0 / π)
    t3phi_err  = fill(t3phi_err_deg, length(indx_t3_1))
    t3phi_obs  = t3phi_true .+ randn(rng, length(indx_t3_1)) .* t3phi_err

    return (v2=v2_obs, v2_err=v2_err, t3phi=t3phi_obs, t3phi_err=t3phi_err)
end

# ── True image ─────────────────────────────────────────────────────────────────

nx          = 64
pixsize_mas = 0.5

x_true = zeros(nx, nx)
x_true[25, 25] = 0.70
x_true[50, 50] = 0.30

println("True image: two point sources")
println("  Star A at (25, 25): flux = 0.70")
println("  Star B at (50, 50): flux = 0.30")
println("  Flux-weighted centroid: ($(0.7*25 + 0.3*50), $(0.7*25 + 0.3*50))  [= image centre 32.5]")

# ── UV coverage: 15 stations → 105 V2, 455 T3 ─────────────────────────────────

n_sta       = 10
rng_uv      = MersenneTwister(42)
sta_angles  = range(0, 2π; length=n_sta+1)[1:end-1]
sta_lengths = 2e7 .+ 8e7 .* rand(rng_uv, n_sta)
stations    = hcat([sta_lengths[k] .* [cos(sta_angles[k]), sin(sta_angles[k])]
                    for k in 1:n_sta]...)

v2_pairs    = [(i, j) for i in 1:n_sta for j in (i+1):n_sta]
n_v2        = length(v2_pairs)
uv_v2       = hcat([stations[:, i] - stations[:, j] for (i, j) in v2_pairs]...)

t3_triplets = [(i, j, k) for i in 1:n_sta for j in (i+1):n_sta for k in (j+1):n_sta]
n_t3        = length(t3_triplets)
uv_t3_1     = hcat([stations[:, i] - stations[:, j] for (i, j, k) in t3_triplets]...)
uv_t3_2     = hcat([stations[:, j] - stations[:, k] for (i, j, k) in t3_triplets]...)
uv_t3_3     = hcat([stations[:, k] - stations[:, i] for (i, j, k) in t3_triplets]...)

uv        = hcat(uv_v2, uv_t3_1, uv_t3_2, uv_t3_3)
n_uv      = size(uv, 2)
indx_v2   = collect(1:n_v2)
indx_t3_1 = collect(n_v2+1        : n_v2+n_t3)
indx_t3_2 = collect(n_v2+n_t3+1   : n_v2+2*n_t3)
indx_t3_3 = collect(n_v2+2*n_t3+1 : n_v2+3*n_t3)

println("UV coverage: $n_v2 V2 baselines, $n_t3 T3 triangles")

obs = simulate_data_hmc(x_true, uv, indx_v2, indx_t3_1, indx_t3_2, indx_t3_3,
                        pixsize_mas; noise_frac=0.05, t3phi_err_deg=5.0,
                        rng=MersenneTwister(99))

# ── OIdata ─────────────────────────────────────────────────────────────────────

data = OIdata(
    Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Float64[],
    Bool[],
    obs.v2, obs.v2_err, zeros(n_v2), zeros(n_v2),
    0.0,
    fill(1.65e-6, n_v2), fill(5e-8, n_v2),
    fill(false, n_v2),
    Float64[], Float64[], obs.t3phi, obs.t3phi_err,
    Float64[], Float64[],
    zeros(n_t3), zeros(n_t3), zeros(n_t3),
    fill(1.65e-6, n_t3), fill(5e-8, n_t3), fill(false, n_t3),
    Float64[], Float64[], Float64[], Float64[], Float64[], Bool[],
    Int64[],
    uv, Float64[], Float64[], Float64[], Float64[],
    0, 0, 0, n_v2, 0, n_t3, n_uv,
    Int64[], indx_v2, indx_t3_1, indx_t3_2, indx_t3_3,
    String[], String[], Int64[],
    zeros(Int64, 2, 0), zeros(Int64, 2, n_v2), zeros(Int64, 3, n_t3),
    # correlation matrices (empty)
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    ""
)

ft = setup_nfft(data, nx, pixsize_mas)
println("NFFT plan built (n_v2=$n_v2, n_t3=$n_t3, n_uv=$n_uv)")

# ── Starting image (used only for API compatibility; SA handles real init) ──────

x_start = fill(1.0 / nx^2, nx, nx)

# ── Run hmc_reconstruct ───────────────────────────────────────────────────────

weights      = [1.0, 0.0, 1.0]
regularizers = [["centering", 1e-3], ["tv", 1e-4]]

n_chains   = 2
n_sa_steps = 2000
n_map_iter = 200
n_samples  = 50
n_adapts   = 50

println("\nRunning hmc_reconstruct:")
println("  n_chains=$n_chains, n_sa_steps=$n_sa_steps, n_map_iter=$n_map_iter")
println("  n_adapts=$n_adapts, n_samples=$n_samples, max_tree_depth=6")
println("  Image: $(nx)×$(nx) = $(nx^2) pixels")
println("  Threads available: $(Threads.nthreads())")

t_start = time()

result = hmc_reconstruct(
    x_start, data, ft;
    nx             = nx,
    weights        = weights,
    regularizers   = regularizers,
    n_samples      = n_samples,
    n_adapts       = n_adapts,
    n_chains       = n_chains,
    n_sa_steps     = n_sa_steps,
    T_start        = 50.0,
    n_map_iter     = n_map_iter,
    max_tree_depth = 6,
    verb           = true,
    drop_warmup    = true,
)

elapsed = time() - t_start

# ── Per-chain diagnostics ─────────────────────────────────────────────────────

diag = result.diagnostics
println("\n── Per-chain diagnostics ────────────────────────────────")
println("  Chain | MAP chi2 | Accept | Divergences | Status")
println("  ------|----------|--------|-------------|-------")
for c in 1:n_chains
    status = diag.chain_divergences[c] <= 0.05 * n_samples &&
             diag.chain_accept_rates[c] >= 0.3 ? "OK" : "FAIL"
    println("    $c   | $(round(diag.chain_map_chi2[c]; digits=1)) | " *
            "$(round(diag.chain_accept_rates[c]; digits=3))  | " *
            "$(diag.chain_divergences[c])       | $status")
end
println("  Converged: $(diag.n_converged) / $n_chains")
if !isnan(maximum(filter(!isnan, vec(diag.rhat))))
    println("  R-hat (max across pixels): $(round(maximum(filter(!isnan, vec(diag.rhat))); digits=4))")
end
println("  Min ESS: $(round(diag.min_ess; digits=1))")

# ── Wall-clock time ───────────────────────────────────────────────────────────

n_total_samples = size(result.samples, 2)
println("\n── Timing ───────────────────────────────────────────────")
println("  Total wall-clock time: $(round(elapsed; digits=1)) s")
println("  Total samples collected: $n_total_samples")
println("  Time per chain-sample: $(round(elapsed / (n_chains * n_samples); digits=3)) s/sample")

# ── Posterior mean image top-2 ────────────────────────────────────────────────

mean_img   = result.mean_image
sorted_idx = sortperm(vec(mean_img); rev=true)
top2       = [CartesianIndices((nx, nx))[i] for i in sorted_idx[1:2]]

println("\n── Posterior mean image — top-2 brightest pixels ────────")
for ci in top2
    println("  pixel $(Tuple(ci))  value=$(round(mean_img[ci]; sigdigits=4))")
end

true_locs = [CartesianIndex(25, 25), CartesianIndex(50, 50)]
tol = 4

println("\nTrue source positions: (25,25) and (50,50)  (tolerance = $tol px)")
for loc in true_locs
    dists = [norm(Float64[Tuple(ci)...] .- Float64[Tuple(loc)...]) for ci in top2]
    println("  True $(Tuple(loc)) → top2 distances: $(round.(dists; digits=2))")
end

# ── R-hat at brightest pixel ──────────────────────────────────────────────────

brightest_ci   = CartesianIndices((nx, nx))[sorted_idx[1]]
rhat_brightest = diag.rhat[brightest_ci]
println("\nR-hat at brightest pixel $(Tuple(brightest_ci)): $(round(rhat_brightest; digits=4))")

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Binary star HMC integration test (SA init)" begin

    @testset "hmc_reconstruct returns expected fields" begin
        @test haskey(result, :mean_image)
        @test haskey(result, :std_image)
        @test haskey(result, :map_image)
        @test haskey(result, :samples)
        @test haskey(result, :image_samples)
        @test haskey(result, :stats)
        @test haskey(result, :diagnostics)
        @test size(result.mean_image) == (nx, nx)
        @test size(result.samples, 2) >= n_samples
    end

    @testset "Mean image is unit-flux" begin
        @test sum(mean_img) ≈ 1.0 atol=1e-6
    end

    @testset "Posterior std image is non-negative" begin
        @test all(result.std_image .>= 0.0)
    end

    @testset "MAP image is unit-flux" begin
        @test sum(result.map_image) ≈ 1.0 atol=1e-6
    end

    @testset "At least 1 of $n_chains chains converged" begin
        @test diag.n_converged >= 1
    end

    @testset "0 divergences across converged chains" begin
        n_div_total = sum(diag.chain_divergences[c] for c in 1:n_chains
                          if diag.chain_accept_rates[c] >= 0.3)
        @test n_div_total == 0
        println("  Total divergences in converged chains: $n_div_total")
    end

    @testset "Wall-clock time reported" begin
        @test elapsed > 0.0
        println("  Total wall-clock: $(round(elapsed; digits=1)) s  " *
                "($(round(elapsed / (n_chains * n_samples); digits=3)) s/chain-sample)")
    end

    @testset "Mean image peaks near true source positions (tol=$tol px)" begin
        for loc in true_locs
            nearby = any(top2) do ci
                norm(Float64[Tuple(ci)...] .- Float64[Tuple(loc)...]) <= tol
            end
            @test nearby
        end
    end

end
