# test/test_sa.jl
#
# Minimal integration test for sa_init.
#
# Setup:
#   - 64×64 image, two point sources at (28,36) and (36,28), flux ratio 0.7/0.3
#   - 10-station circular array → 45 V2, 120 T3 triangles
#   - 5% relative V2 noise, 5° T3phi noise
#   - weights [1.0, 0.0, 1.0], regularizers: centering 1e-3, tv 1e-4
#   - n_steps=2000, T_start=50.0
#   - Assert: chi2 of returned image < 2000
#   - Report wall-clock time

using OIImage
using Test
using Random
using LinearAlgebra

Random.seed!(1234)

# ── True image ─────────────────────────────────────────────────────────────────
nx          = 64
pixsize_mas = 0.5

x_true = zeros(nx, nx)
x_true[28, 36] = 0.70
x_true[36, 28] = 0.30

println("True image: two point sources")
println("  Star A at (28, 36): flux = 0.70")
println("  Star B at (36, 28): flux = 0.30")

# ── UV coverage: 10 stations → 45 V2, 120 T3 ──────────────────────────────────
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

# ── Simulate observables ───────────────────────────────────────────────────────
rng_obs = MersenneTwister(99)
x_norm  = x_true ./ sum(x_true)
dft     = setup_dft(uv, nx, pixsize_mas)
cvis    = dft * vec(x_norm)

v2_true = abs2.(cvis[indx_v2])
v2_err  = 0.05 .* max.(v2_true, 1e-6)
v2_obs  = v2_true .+ randn(rng_obs, n_v2) .* v2_err

t3_true    = cvis[indx_t3_1] .* cvis[indx_t3_2] .* cvis[indx_t3_3]
t3phi_true = angle.(t3_true) .* (180.0 / π)
t3phi_err  = fill(5.0, n_t3)
t3phi_obs  = t3phi_true .+ randn(rng_obs, n_t3) .* t3phi_err

# ── OIdata ─────────────────────────────────────────────────────────────────────
data = OIdata(
    # vis (empty)
    Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Float64[],
    Bool[],
    # V2
    v2_obs, v2_err, zeros(n_v2), zeros(n_v2),
    0.0,
    fill(1.65e-6, n_v2), fill(5e-8, n_v2),
    fill(false, n_v2),
    # T3 (t3amp empty, t3phi in degrees)
    Float64[], Float64[], t3phi_obs, t3phi_err,
    Float64[], Float64[],
    zeros(n_t3), zeros(n_t3), zeros(n_t3),
    fill(1.65e-6, n_t3), fill(5e-8, n_t3), fill(false, n_t3),
    # Flux (empty)
    Float64[], Float64[], Float64[], Float64[], Float64[], Bool[],
    Int64[],
    # UV
    uv, Float64[], Float64[], Float64[], Float64[],
    # counts
    0, 0, 0, n_v2, 0, n_t3, n_uv,
    # index arrays
    Int64[], indx_v2, indx_t3_1, indx_t3_2, indx_t3_3,
    # station metadata
    String[], String[], Int64[],
    zeros(Int64, 2, 0), zeros(Int64, 2, n_v2), zeros(Int64, 3, n_t3),
    # correlation matrices (empty)
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    spzeros_empty(), Int64[], spzeros_empty(), Int64[],
    ""
)

ft = setup_nfft(data, nx, pixsize_mas)
println("NFFT plan built")

# ── Compute reference chi2 at true image ──────────────────────────────────────
g_ref = zeros(nx, nx)
chi2_true = chi2_fg(x_norm, g_ref, ft, data; weights=[1.0, 0.0, 1.0], verb=false)
println("chi2(x_true) = $(round(chi2_true; digits=1))  " *
        "($(n_v2 + n_t3) data points, chi2_red=$(round(chi2_true/(n_v2+n_t3); digits=3)))")

# ── Run sa_init ───────────────────────────────────────────────────────────────
weights      = [1.0, 0.0, 1.0]
regularizers = [["centering", 1e-3], ["tv", 1e-4]]

println("\nRunning sa_init: n_steps=5000, T_start=50.0, step_size=0.1")

t_start = time()
x_sa = sa_init(data, ft, nx;
               regularizers = regularizers,
               weights      = weights,
               n_steps      = 5000,
               T_start      = 50.0,
               T_end        = 1.0,
               step_size    = 0.1,
               rng          = MersenneTwister(7),
               verb         = true)
elapsed = time() - t_start

# ── Report results ─────────────────────────────────────────────────────────────
g_sa   = zeros(nx, nx)
chi2_sa = chi2_fg(x_sa, g_sa, ft, data; weights=weights, verb=false)
chi2_red_sa = chi2_sa / (n_v2 + n_t3)

println("\n── SA result ────────────────────────────────────────────")
println("  Wall-clock time  : $(round(elapsed; digits=2)) s")
println("  chi2(x_sa)       : $(round(chi2_sa; digits=1))")
println("  chi2_reduced     : $(round(chi2_red_sa; digits=3))")
println("  flux             : $(round(sum(x_sa); digits=6))")

sorted_idx = sortperm(vec(x_sa); rev=true)
top2 = [CartesianIndices((nx, nx))[i] for i in sorted_idx[1:2]]
println("  Top-2 pixels     : $(Tuple(top2[1])) val=$(round(x_sa[top2[1]]; sigdigits=3))  " *
        "$(Tuple(top2[2])) val=$(round(x_sa[top2[2]]; sigdigits=3))")
println("  True positions   : (28, 36) and (36, 28)")

# ── Tests ─────────────────────────────────────────────────────────────────────
@testset "sa_init: binary star smoke test" begin

    @testset "returns normalized image" begin
        @test size(x_sa) == (nx, nx)
        @test sum(x_sa) ≈ 1.0 atol=1e-10
        @test all(x_sa .>= 0.0)
    end

    @testset "chi2 < 2000 (well below random baseline)" begin
        @test chi2_sa < 2000
        println("  chi2=$(round(chi2_sa; digits=1))  (threshold: 2000,  " *
                "chi2_true=$(round(chi2_true; digits=1)))")
    end

    @testset "wall-clock time reported" begin
        @test elapsed > 0.0
        println("  SA timing: $(round(elapsed; digits=2)) s for 5000 steps " *
                "($(round(1000*elapsed/5000; digits=1)) ms/step)")
    end

end
