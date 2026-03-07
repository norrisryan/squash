# sa_init.jl
#
# Simulated annealing initialization for interferometric image reconstruction.
# Runs multiple independent SA restarts and returns the best-ever image.
#
# Each restart uses sign-gradient Metropolis in log-space:
#   y_new = y - lr * step_size * sign(gy) + step_size * randn()
# The sign update gives every pixel a constant-magnitude gradient step regardless
# of gradient magnitude, allowing background pixels to accumulate flux at source
# positions as quickly as the dominant central pixel loses flux.
#
# Temperature in chi²_reduced units: T_abs = T * n_data.

"""
    sa_init(data, ft, nx; kwargs...) -> Matrix{Float64}

Simulated annealing initialization for interferometric image reconstruction.
Runs `n_restarts` independent SA chains (sign-gradient Metropolis in log-space)
over the total `n_steps` budget and returns the best-ever normalized nx×nx image.

Temperature is in chi²_reduced units: T_start=50 allows uphill moves of
50 chi²/n_data at the start, matching the OITOOLS chi²_reduced convention.

# Keyword arguments
- `regularizers`      : regularizer list (same format as `crit_fg`)
- `weights`           : `[w_v2, w_t3amp, w_t3phi]` chi² weights (default: `[1,0,1]`)
- `n_steps::Int`      : total SA steps across all restarts (default: 5000)
- `n_restarts::Int`   : number of independent SA restarts (default: 3)
- `T_start::Float64`  : initial temperature in chi²_reduced units (default: 50.0)
- `T_end::Float64`    : final temperature in chi²_reduced units (default: 1.0)
- `step_size::Float64`: log-space step size per pixel per step (default: 0.1)
- `lr::Float64`       : gradient vs noise balance; 0=pure noise, 1=balanced (default: 1.0)
- `rng`               : random number generator (default: `Random.default_rng()`)
- `verb::Bool`        : print per-restart progress (default: false)
"""
function sa_init(
    data, ft, nx::Int;
    regularizers   = [],
    weights        = [1.0, 0.0, 1.0],
    n_steps::Int   = 5000,
    n_restarts::Int = 3,
    T_start::Float64 = 50.0,
    T_end::Float64   = 1.0,
    step_size::Float64 = 0.1,
    lr::Float64 = 1.0,
    rng = Random.default_rng(),
    verb::Bool = false,
)
    w = Float64.(weights)
    n_data = (w[1] > 0) * data.nv2 + (w[2] > 0) * data.nt3amp + (w[3] > 0) * data.nt3phi
    n_data = max(n_data, 1)

    n_steps_per_restart = max(n_steps ÷ n_restarts, 1)

    global_best_E = Inf
    global_best_x = zeros(nx, nx)
    global_best_x[nx÷2+1, nx÷2+1] = 1.0   # fallback if all restarts fail

    # ── Adaptive centering weight ─────────────────────────────────────────────
    # Scale centering to ~10% of initial chi2 when centroid is nx/4 px off-centre.
    # reg_centering returns (displacement_pixels)², so weight = 0.1*chi2_0 / (nx/4)².
    # This makes centering meaningful at all SA temperature scales without user-tuning.
    x0_tmp = fill(0.5 / nx^2, nx, nx)
    x0_tmp[nx÷2+1, nx÷2+1] += 0.5
    x0_tmp ./= sum(x0_tmp)
    g0_tmp = zeros(nx, nx)
    chi2_0 = chi2_fg(x0_tmp, g0_tmp, ft, data; weights=w, verb=false)
    adaptive_centering_weight = 0.1 * chi2_0 / (nx / 4)^2
    centering_regs = vcat(regularizers, [["centering", adaptive_centering_weight]])

    # ── Combined cost + gradient ──────────────────────────────────────────────
    function crit_fg_val!(img, g)
        fill!(g, 0.0)
        c2  = chi2_fg(img, g, ft, data; weights=w, verb=false)
        reg = extended_regularization(img, g; regularizers=centering_regs, verb=false)
        return c2 + reg
    end

    for restart in 1:n_restarts
        rng_r = MersenneTwister(rand(rng, UInt32))

        # ── Starting image: single bright central pixel + uniform background ──
        x = fill(0.5 / nx^2, nx, nx)
        x[nx÷2+1, nx÷2+1] += 0.5
        x ./= sum(x)
        y = log.(x)

        g  = zeros(nx, nx)
        E  = crit_fg_val!(x, g)

        # Sign-gradient: ∂cost/∂y_i = x_i * (g_i - ⟨x,g⟩), then take sign
        function sign_gy(x_img, g_img)
            gdotx = dot(vec(x_img), vec(g_img))
            return sign.(x_img .* (g_img .- gdotx))
        end
        sgy = sign_gy(x, g)

        best_E = E
        best_y = copy(y)
        n_accept_window = 0
        window = 500

        verb && println("SA restart $restart/$n_restarts : " *
                        "E_init=$(round(E; digits=1))  " *
                        "chi²_red=$(round(E/n_data; digits=2))")

        for step in 1:n_steps_per_restart
            T_red = T_start * (T_end / T_start)^(step / n_steps_per_restart)
            T_abs = T_red * n_data

            # Sign-gradient proposal: each pixel steps by ±step_size in gradient direction
            y_new = y .- (lr * step_size) .* sgy .+ step_size .* randn(rng_r, nx, nx)

            y_shift = maximum(y_new)
            x_new   = exp.(y_new .- y_shift)
            x_new ./= sum(x_new)

            E_new = crit_fg_val!(x_new, g)
            ΔE    = E_new - E

            if ΔE < 0.0 || rand(rng_r) < exp(-ΔE / T_abs)
                y   = y_new
                x   = x_new
                E   = E_new
                sgy = sign_gy(x, g)   # update sign-gradient at accepted state
                n_accept_window += 1
                if E < best_E
                    best_E = E
                    best_y = copy(y)
                end
            end

            if verb && step % window == 0
                println("  [r$restart] step $step/$n_steps_per_restart : " *
                        "T_red=$(round(T_red; digits=2))  best=$(round(best_E; digits=1))  " *
                        "chi²_red=$(round(best_E/n_data; digits=2))  " *
                        "accept=$(round(n_accept_window/window; digits=3))")
                n_accept_window = 0
            end
        end

        verb && println("  restart $restart done: best_E=$(round(best_E; digits=1))  " *
                        "chi²_red=$(round(best_E/n_data; digits=2))")

        if best_E < global_best_E
            global_best_E = best_E
            y_shift = maximum(best_y)
            x_best  = exp.(best_y .- y_shift)
            global_best_x = x_best ./ sum(x_best)
        end
    end

    verb && println("SA done ($(n_restarts) restarts): " *
                    "global_best_E=$(round(global_best_E; digits=1))  " *
                    "chi²_red=$(round(global_best_E/n_data; digits=2))")

    return global_best_x
end
