# test/test_hmc.jl
#
# Synthetic binary-star integration test for hmc_reconstruct.
#
# What this does:
#   1. Build a 64×64 true image with two point sources (binary star).
#   2. Simulate squared-visibility (V2) observables on a realistic UV grid.
#   3. Run hmc_reconstruct with 500 HMC samples.
#   4. Check that the posterior mean image peaks near the true source positions.
#   5. Report wall-clock time.
#
# Run with:
#   julia --project test/test_hmc.jl

using OIImage
using Test
using Random
using Statistics
using LinearAlgebra

Random.seed!(1234)

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
    simulate_v2(x_true, uv, pixsize; snr=10.0, rng=Random.default_rng())

Compute noiseless V2 from a true image and add Gaussian noise.

Returns (v2_obs, v2_err, uv).
"""
function simulate_v2(x_true::Matrix{Float64}, uv::Matrix{Float64},
                     pixsize::Float64; snr::Float64=10.0,
                     rng=Random.default_rng())
    nx = size(x_true, 1)
    n_uv = size(uv, 2)

    # Analytic DFT of the image at each UV point.
    # uv is in cycles/rad, pixsize in rad.
    # Spatial frequency in cycles/pixel: u_pix = u_cyc_per_rad * pixsize
    v2_true = zeros(n_uv)
    for k in 1:n_uv
        u_pix = uv[1, k] * pixsize
        v_pix = uv[2, k] * pixsize
        vis = 0.0 + 0.0im
        for j in 1:nx, i in 1:nx
            phase = 2π * (u_pix * (i - nx/2 - 1) / nx + v_pix * (j - nx/2 - 1) / nx)
            vis += x_true[i, j] * exp(-1im * phase)
        end
        v2_true[k] = abs2(vis)
    end

    # Add noise
    v2_err = v2_true ./ snr
    v2_err = max.(v2_err, 1e-6)
    v2_obs = v2_true .+ randn(rng, n_uv) .* v2_err

    return v2_obs, v2_err
end

# ── Build true image ──────────────────────────────────────────────────────────

nx = 64
pixsize = 0.5e-3 * (π / 180 / 3600)  # 0.5 mas in radians

x_true = zeros(nx, nx)
# Star A: 70% flux at pixel (20,20)   (0-indexed centre = nx/2)
# Star B: 30% flux at pixel (44,44)
x_true[20, 20] = 0.70
x_true[44, 44] = 0.30

println("True image: two point sources at (20,20) and (44,44)")
println("  Peak values: $(x_true[20,20]) and $(x_true[44,44])")

# ── Simulate UV coverage ──────────────────────────────────────────────────────
# Circular UV coverage, ~150 baselines up to 100 Mλ

n_uv = 150
angles = range(0, 2π; length=n_uv+1)[1:end-1]
baseline_lengths = 1e7 .+ (1e8 - 1e7) .* rand(MersenneTwister(42), n_uv)  # 10–100 Mλ
uv = hcat([baseline_lengths[k] .* [cos(angles[k]), sin(angles[k])]
           for k in 1:n_uv]...)   # 2 × n_uv

v2_obs, v2_err = simulate_v2(x_true, uv, pixsize; snr=15.0)

println("Simulated $n_uv V2 observables, SNR≈15")

# ── Build OIdata ──────────────────────────────────────────────────────────────
# We only populate V2 fields; all T3 and differential-vis fields are empty.

data = OIdata(
    # vis (empty): visamp, visamp_err, visphi, visphi_err,
    #              vis_baseline, vis_mjd, vis_lam, vis_dlam, vis_flag
    Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Float64[],
    Bool[],
    # V2: v2, v2_err, v2_baseline, v2_mjd, mean_mjd,
    #     v2_lam, v2_dlam, v2_flag
    v2_obs, v2_err, zeros(n_uv), zeros(n_uv),
    0.0,
    fill(1.65e-6, n_uv), fill(5e-8, n_uv),
    fill(false, n_uv),
    # T3 (empty): t3amp, t3amp_err, t3phi, t3phi_err,
    #             t3phi_vonmises_err, t3phi_vonmises_chi2_offset,
    #             t3_baseline, t3_maxbaseline, t3_mjd,
    #             t3_lam, t3_dlam, t3_flag
    Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[],
    Float64[], Float64[], Float64[],
    Float64[], Float64[], Bool[],
    # Flux (empty): flux, flux_err, flux_mjd, flux_lam, flux_dlam,
    #               flux_flag, flux_sta_index
    Float64[], Float64[], Float64[], Float64[], Float64[], Bool[],
    Int64[],
    # UV: uv, uv_lam, uv_dlam, uv_mjd, uv_baseline
    uv, Float64[], Float64[], Float64[], Float64[],
    # counts: nflux, nvisamp, nvisphi, nv2, nt3amp, nt3phi, nuv
    0, 0, 0, n_uv, 0, 0, n_uv,
    # index arrays: indx_vis, indx_v2, indx_t3_1, indx_t3_2, indx_t3_3
    Int64[], collect(1:n_uv), Int64[], Int64[], Int64[],
    # station metadata: sta_name, tel_name, sta_index,
    #                   vis_sta_index, v2_sta_index, t3_sta_index, filename
    String[], String[], Int64[],
    zeros(Int64, 2, 0), zeros(Int64, 2, n_uv), zeros(Int64, 3, 0),
    ""
)

# ── Setup NFFT ────────────────────────────────────────────────────────────────

ft = setup_nfft(data, nx, pixsize)
println("NFFT plan built")

# ── Flat starting image ───────────────────────────────────────────────────────

x_start = fill(1.0 / nx^2, nx, nx)

# ── Run hmc_reconstruct ───────────────────────────────────────────────────────

n_samples = 500
n_adapts  = 200

println("\nRunning hmc_reconstruct: $n_adapts adapts + $n_samples samples on $(nx)×$(nx) image...")
t_start = time()

result = hmc_reconstruct(
    x_start, data, ft;
    nx           = nx,
    weights      = [1.0, 0.0, 0.0],   # V2 only (chi2_fg order: [w_v2, w_t3amp, w_t3phi])
    regularizers = [["laplacian", 1e-4]],
    n_samples    = n_samples,
    n_adapts     = n_adapts,
    n_pathfinder_runs  = 2,
    n_pathfinder_draws = 100,
    verb         = true,
    drop_warmup  = true,
)

elapsed = time() - t_start
println("\nWall-clock time: $(round(elapsed; digits=1)) s for $n_samples HMC samples")
println("= $(round(elapsed / n_samples; digits=3)) s/sample")

# ── Check results ─────────────────────────────────────────────────────────────

mean_img = result.mean_image

# The two brightest pixels in the mean image should be near (20,20) and (44,44).
# We allow a tolerance of ±3 pixels.
sorted_idx = sortperm(vec(mean_img); rev=true)
top2 = [CartesianIndices((nx, nx))[i] for i in sorted_idx[1:2]]

println("\nPosterior mean image — top-2 brightest pixels:")
for ci in top2
    println("  pixel $(Tuple(ci))  value=$(round(mean_img[ci]; sigdigits=4))")
end

true_locs = [CartesianIndex(20, 20), CartesianIndex(44, 44)]
tol = 3

@testset "Binary star HMC integration test" begin

    @testset "hmc_reconstruct returns expected fields" begin
        @test haskey(result, :mean_image)
        @test haskey(result, :std_image)
        @test haskey(result, :map_image)
        @test haskey(result, :samples)
        @test haskey(result, :image_samples)
        @test size(result.mean_image) == (nx, nx)
        @test size(result.samples, 2) == n_samples
    end

    @testset "Mean image is unit-flux" begin
        @test sum(mean_img) ≈ 1.0 atol=1e-6
    end

    @testset "Mean image peaks near true source positions" begin
        # For each true source, at least one of the top-2 peaks is within tol pixels.
        for loc in true_locs
            nearby = any(top2) do ci
                norm(Float64[Tuple(ci)...] .- Float64[Tuple(loc)...]) <= tol
            end
            @test nearby
        end
    end

    @testset "Posterior std image is non-negative" begin
        @test all(result.std_image .>= 0.0)
    end

    @testset "MAP image is unit-flux" begin
        @test sum(result.map_image) ≈ 1.0 atol=1e-6
    end

    @testset "Wall-clock time reported" begin
        # Just a sanity check that we got a positive elapsed time
        @test elapsed > 0.0
    end
end
