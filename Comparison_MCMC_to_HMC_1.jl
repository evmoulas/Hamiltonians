#=
Comparison_MCMC_to_HMC_1 – random-walk Metropolis and HMC side by side (y = a + k·x)
===================================================================================

Both samplers get the same data, the same prior, the same starting point and the
same number of iterations, and run next to each other:

  Metropolis (random walk, Introduction_MCMC_3):  θ′ ~ N(θ, s²·C),
      α = min(1, p(θ′|d)/p(θ|d))                  – 1 model evaluation per iteration
  HMC (Introduction_HMC_2):  p ~ N(0, M), L leapfrog steps of size ε,
      α = min(1, e^(−ΔH))                         – L + 1 gradient evaluations per iteration

Problem: y = a + k·x, priors k ~ N(0.5, 2²), a ~ N(0, 5²) (weak), θ = (k, a).

"Who converges faster?" depends on what is counted:
  * per ITERATION, HMC wins clearly: one trajectory travels far and is almost
    always accepted, so successive samples are nearly independent;
  * per MODEL EVALUATION (the computing cost) an HMC iteration is L + 1 times as
    expensive. For two parameters the two samplers are then of SIMILAR
    efficiency (measured with this program: ESS per 1000 evaluations ≈ 23
    (Metropolis) versus 20 – 62 (HMC) without preconditioning, 140 – 145 versus
    110 – 230 with it – depending on the tuning and on the random run, either can
    win). HMC's advantage grows with the number of parameters d: a random
    walk needs a step ∝ d^(−1/2) to keep being accepted, HMC only ∝ d^(−1/4) –
    with hundreds of parameters HMC wins by orders of magnitude (not shown here).
Toggle "x-axis: model evaluations" to switch the point of view, and "equal cost"
to give Metropolis L + 1 iterations for every HMC iteration.

"Precondition both" uses the posterior covariance Σ for both samplers
(Metropolis: proposal s²Σ, best s ≈ 1.7; HMC: mass matrix M = Σ⁻¹, best L·ε ≈ π/2),
as if Σ had been learned in a pilot run (here the exact Σ is used). This removes
the correlation between k and a for both and compares them at their best.

Convergence measures
  * error of the running mean (after burn-in) in units of the posterior:
        d_n = √( (m̂_n − m)ᵀ Σ⁻¹ (m̂_n − m) )        (m, Σ: exact posterior)
    For independent samples d_n ≈ √(2/n) (dashed line); a chain with an
    effective sample size ESS behaves like √(2/ESS).
  * "reached the posterior": first iteration inside the 99.7 % ellipse.
  * ESS (batch means) and ESS per 1000 model evaluations.

Run from a terminal:      julia Comparison_MCMC_to_HMC_1.jl
or from the Julia REPL:   include("Comparison_MCMC_to_HMC_1.jl")
=#

using Pkg
Pkg.activate(@__DIR__)

using GLMakie
using Printf
using Random

# =============================================================================
# USER INPUT
# =============================================================================
const XLIM     = (0.0, 10.0)        # x range of the data
const KLIM     = (-1.0, 4.0)        # k range of the misfit maps (full view)
const ALIM     = (-8.0, 8.0)        # a range of the misfit maps (full view)
const NG       = 161                # grid (NG × NG) of the misfit maps
const NMAX     = 50                 # largest number of data points
const SEED     = 1                  # seed of the first noise realisation

const K_TRUE   = 1.5                # default true slope
const A_TRUE   = 2.0                # default true intercept
const SIGMA    = 2.0                # default noise standard deviation
const NDATA    = 15                 # default number of data points
const START    = (-0.5, -5.0)       # default starting point (k₀, a₀)
const PRIOR    = (0.5, 2.0, 0.0, 5.0)   # prior (μ_k, σ_k, μ_a, σ_a)
const STEP     = 0.3                # default Metropolis step s
const EPS      = 0.05               # default leapfrog step ε
const NLEAP    = 20                 # default number of leapfrog steps L
const BATCHES  = [1, 10, 100, 1000] # iterations per press of "Run"
const NPATH    = 100                # number of last steps drawn as a path
const DCHI     = [2.30, 6.18, 11.8] # Δχ² of the 68 / 95 / 99.7 % regions (2 parameters)
const DIVERGE  = 1000.0             # ΔH above this: divergent trajectory
const NPLOT    = 1500               # points of the (log-spaced) convergence curves

const C_MH    = :darkorange2        # Metropolis
const C_HMC   = :dodgerblue3        # HMC
const C_DATA  = :black
const C_TRUE  = :gray40
const C_POST  = :black
const C_MIS   = :purple4

# =============================================================================
# MATHEMATICS (independent of the plotting)
# =============================================================================
"Symmetric 2×2 matrix [xx xy; xy yy]; index 1 = k, 2 = a."
struct Sym2
    xx::Float64
    xy::Float64
    yy::Float64
end
const I2 = Sym2(1.0, 0.0, 1.0)
Base.:*(c::Real, S::Sym2) = Sym2(c * S.xx, c * S.xy, c * S.yy)
det2(S::Sym2) = S.xx * S.yy - S.xy^2
inv2(S::Sym2) = (1 / det2(S)) * Sym2(S.yy, -S.xy, S.xx)
mul(S::Sym2, v) = (S.xx * v[1] + S.xy * v[2], S.xy * v[1] + S.yy * v[2])
quad(S::Sym2, v) = v[1] * (S.xx * v[1] + S.xy * v[2]) + v[2] * (S.xy * v[1] + S.yy * v[2])
function chol2(S::Sym2)
    l11 = sqrt(S.xx); l21 = S.xy / l11
    return l11, l21, sqrt(max(S.yy - l21^2, 0.0))
end
sd2(S::Sym2) = (sqrt(S.xx), sqrt(S.yy))
corr2(S::Sym2) = S.xy / sqrt(S.xx * S.yy)
"Eigenvalues of A·B for symmetric positive definite 2×2 A, B."
function eig_prod(A::Sym2, B::Sym2)
    m11 = A.xx * B.xx + A.xy * B.xy; m12 = A.xx * B.xy + A.xy * B.yy
    m21 = A.xy * B.xx + A.yy * B.xy; m22 = A.xy * B.xy + A.yy * B.yy
    t = m11 + m22; d = m11 * m22 - m12 * m21
    r = sqrt(max(t^2 / 4 - d, 0.0))
    return t / 2 - r, t / 2 + r
end

struct DataSums
    n::Float64; x::Float64; xx::Float64; y::Float64; xy::Float64; yy::Float64; s2::Float64
end
DataSums(x, y, σ) = DataSums(length(x), sum(x), sum(abs2, x), sum(y), sum(x .* y), sum(abs2, y), σ^2)
chi2(k, a, S::DataSums) =
    max(S.yy - 2a * S.y - 2k * S.xy + S.n * a^2 + 2a * k * S.x + k^2 * S.xx, 0.0) / S.s2

struct Prior
    mk::Float64; sk::Float64; ma::Float64; sa::Float64
end
logpost(θ, S, P::Prior) = -0.5 * chi2(θ[1], θ[2], S) - 0.5 * ((θ[1] - P.mk) / P.sk)^2 - 0.5 * ((θ[2] - P.ma) / P.sa)^2
gradlogpost(θ, S, P::Prior) = ((S.xy - θ[2] * S.x - θ[1] * S.xx) / S.s2 - (θ[1] - P.mk) / P.sk^2,
                               (S.y - S.n * θ[2] - θ[1] * S.x) / S.s2 - (θ[2] - P.ma) / P.sa^2)
function exact_posterior(S::DataSums, P::Prior)
    Λ = Sym2(S.xx / S.s2 + 1 / P.sk^2, S.x / S.s2, S.n / S.s2 + 1 / P.sa^2)
    C = inv2(Λ)
    return mul(C, (S.xy / S.s2 + P.mk / P.sk^2, S.y / S.s2 + P.ma / P.sa^2)), C
end
function lsq(S::DataSums)
    C = inv2(Sym2(S.xx / S.s2, S.x / S.s2, S.n / S.s2))
    return mul(C, (S.xy / S.s2, S.y / S.s2)), C
end

function ellipse(m, C::Sym2, r; n = 120)
    l11, l21, l22 = chol2(C)
    pts = [Point2f(m[1] + r * l11 * cos(t), m[2] + r * (l21 * cos(t) + l22 * sin(t))) for t in range(0, 2π, length = n)]
    push!(pts, Point2f(NaN, NaN))
    return pts
end

# ---------------------------------------------------------------- Metropolis
"One random-walk Metropolis step θ′ ~ N(θ, s²C). Returns (θ_new, lp_new, accepted)."
function mh_step(rng, θ, lp, s, C::Sym2, S, P)
    l11, l21, l22 = chol2(C)
    z1, z2 = randn(rng), randn(rng)
    θp  = (θ[1] + s * l11 * z1, θ[2] + s * (l21 * z1 + l22 * z2))
    lpp = logpost(θp, S, P)
    acc = log(rand(rng)) < lpp - lp
    return acc ? (θp, lpp, true) : (θ, lp, false)
end

# ---------------------------------------------------------------- HMC
"One HMC iteration (L leapfrog steps, mass matrix M). Returns (θ_new, accepted, divergent, trajectory)."
function hmc_step(rng, θ, ε, L, M::Sym2, Minv::Sym2, S, P)
    U(t)  = -logpost(t, S, P)
    dU(t) = (g = gradlogpost(t, S, P); (-g[1], -g[2]))
    l11, l21, l22 = chol2(M)
    z1, z2 = randn(rng), randn(rng)
    p0 = (l11 * z1, l21 * z1 + l22 * z2)
    H0 = U(θ) + 0.5 * quad(Minv, p0)
    traj = [Point2f(θ...)]
    t = θ; g = dU(t)
    ph = (p0[1] - 0.5ε * g[1], p0[2] - 0.5ε * g[2])
    for i in 1:L
        v = mul(Minv, ph)
        t = (t[1] + ε * v[1], t[2] + ε * v[2])
        g = dU(t)
        push!(traj, Point2f(t...))
        if !all(isfinite, t) || maximum(abs, t) > 1e6
            return θ, false, true, traj
        end
        i < L && (ph = (ph[1] - ε * g[1], ph[2] - ε * g[2]))
    end
    p1 = (ph[1] - 0.5ε * g[1], ph[2] - 0.5ε * g[2])
    ΔH = U(t) + 0.5 * quad(Minv, p1) - H0
    div = !(ΔH < DIVERGE)
    acc = !div && log(rand(rng)) < -ΔH
    return acc ? t : θ, acc, div, traj
end

# ---------------------------------------------------------------- statistics
mean_(v) = sum(v) / length(v)
var_(v) = (m = mean_(v); sum(abs2, v .- m) / (length(v) - 1))
"Effective sample size by batch means."
function ess(v)
    n = length(v)
    n < 50 && return NaN
    b = floor(Int, sqrt(n)); nb = n ÷ b
    bm = [mean_(view(v, (i-1)*b+1:i*b)) for i in 1:nb]
    vb = var_(bm); vx = var_(v)
    (vb <= 0 || vx <= 0) && return NaN
    return min(n, n * vx / (b * vb))
end
"Up to n log-spaced indices in 1:N."
logidx(N, n) = N <= n ? collect(1:N) : unique(round.(Int, exp.(range(0, log(N), length = n))))

fmt(v) = isnan(v) ? "–" : @sprintf("%.3g", abs(v) < 1e-9 ? 0.0 : v)

# =============================================================================
# APPLICATION
# =============================================================================
"State of one sampler: visited states, cumulative cost, acceptance, last trajectory."
mutable struct Chain
    ks::Vector{Float64}
    as::Vector{Float64}
    cost::Vector{Float64}          # model evaluations up to each iteration
    nacc::Int
    ndiv::Int
    lp::Float64
    traj::Vector{Point2f}
end
Chain(θ, lp) = Chain([θ[1]], [θ[2]], [0.0], 0, 0, lp, Point2f[])

function build_app()
    # ---------------------------------------------------------------- state
    rng_d  = Ref(Xoshiro(SEED))
    rng_m  = Ref(Xoshiro(SEED + 1000)); rng_h = Ref(Xoshiro(SEED + 2000))
    xall   = Ref(zeros(NMAX)); eall = Ref(zeros(NMAX))
    xd     = Ref(Float64[]);   yd   = Ref(Float64[])
    S      = Ref(DataSums(2, 1, 1, 0, 0, 0, 1))
    P      = Prior(PRIOR...)
    used   = Ref((K_TRUE, A_TRUE, SIGMA, NDATA))
    start  = Ref(START)
    ch     = [Chain(START, 0.0), Chain(START, 0.0)]   # 1 = Metropolis, 2 = HMC

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Comparison: random-walk Metropolis versus Hamiltonian Monte Carlo  (Bayesian fit of  y = a + k·x)",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(520))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")        # no fixed height: it takes the space the controls leave

    maps = right[1, 1] = GridLayout()
    axM = [Axis(maps[1, j]; xlabel = "slope k", ylabel = j == 1 ? "intercept a" : "",
                title = j == 1 ? "Metropolis (random walk)" : "Hamiltonian Monte Carlo", titlecolor = (C_MH, C_HMC)[j])
           for j in 1:2]
    colgap!(maps, 14)
    axT = Axis(right[2, 1]; ylabel = "slope k", title = "trace of k")
    conv = right[3, 1] = GridLayout()
    axE = Axis(conv[1, 1]; xscale = log10, yscale = log10, ylabel = "error of the mean  d_n",
               title = "convergence of the running mean (after burn-in)")
    tab = conv[1, 2] = GridLayout(valign = :center, tellheight = false)
    colgap!(conv, 20)
    rowsize!(right, 1, Auto(1.6)); rowsize!(right, 2, Auto(0.7)); rowsize!(right, 3, Auto(1.0)); rowgap!(right, 6)
    for a in (axD, axM..., axT), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
    end
    for name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(axE)) && deregister_interaction!(axE, name)
    end

    # comparison table
    rows = ["iterations", "model evaluations", "accepted", "reached posterior at", "mean  k,  a", "st.dev.  k,  a",
            "ESS  k,  a", "ESS per 1000 evaluations"]
    Label(tab[1, 2], "Metropolis", color = C_MH, font = :bold, fontsize = 13)
    Label(tab[1, 3], "HMC", color = C_HMC, font = :bold, fontsize = 13)
    cells = Matrix{Label}(undef, length(rows), 2)
    for (i, r) in enumerate(rows)
        Label(tab[i+1, 1], r, halign = :right, fontsize = 12, color = :gray25)
        for j in 1:2
            cells[i, j] = Label(tab[i+1, j+1], "–", fontsize = 12, width = 125)
        end
    end
    exactlab = Label(tab[length(rows)+2, 1:3], " ", fontsize = 12, color = :gray25)
    rowgap!(tab, 1); colgap!(tab, 10)

    # display options
    tg = right[4, 1] = GridLayout(halign = :center, tellwidth = false)
    tg_cost  = Toggle(tg[1, 1], active = false); Label(tg[1, 2], "x-axis: model evaluations", halign = :left)
    tg_zoom  = Toggle(tg[1, 3], active = false); Label(tg[1, 4], "zoom on posterior", halign = :left)
    tg_path  = Toggle(tg[1, 5], active = true);  Label(tg[1, 6], "paths / last trajectory", halign = :left)
    colgap!(tg, 6)
    for c in (2, 4)
        colgap!(tg, c, 18)
    end

    # --- 1. data
    Label(left[2, 1], "1 · data   y = a_true + k_true·x + σ·ε", font = :bold, halign = :left, tellwidth = false)
    sgD = SliderGrid(left[3, 1],
        (label = "k_true", range = 0:0.05:3, startvalue = K_TRUE, format = "{:.2f}"),
        (label = "a_true", range = -5:0.1:5, startvalue = A_TRUE, format = "{:.1f}"),
        (label = "noise σ", range = 0.1:0.1:5, startvalue = SIGMA, format = "{:.1f}"),
        (label = "N points", range = 3:NMAX, startvalue = NDATA, format = "{:d}");
        tellheight = true, width = 480, halign = :left)
    sl_kt, sl_at, sl_sig, sl_n = sgD.sliders
    bD = left[4, 1] = GridLayout(halign = :left)
    btn_init  = Button(bD[1, 1], label = "Initialise data", buttoncolor = RGBf(0.80, 0.88, 1.0), height = 28)
    btn_noise = Button(bD[1, 2], label = "new noise realisation", height = 28)
    startlabel = Label(left[5, 1], " ", halign = :left, tellwidth = false, color = :gray30, fontsize = 12,
                       justification = :left)

    # --- 2. samplers
    nice = sort(unique(round.(vcat([m .* 10.0^e for e in -3:1 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3)))
    Label(left[6, 1], rich(rich("2 · Metropolis", color = C_MH), "  θ′ ~ N(θ, s²C)   and   ",
                           rich("HMC", color = C_HMC), "  (ε, L, mass M)"), font = :bold, halign = :left, tellwidth = false)
    sgS = SliderGrid(left[7, 1],
        (label = "Metropolis step s", range = filter(v -> 0.01 <= v <= 5, nice), startvalue = STEP, format = "{:.3g}"),
        (label = "HMC leapfrog step ε", range = filter(v -> 0.001 <= v <= 3, nice), startvalue = EPS, format = "{:.3g}"),
        (label = "HMC leapfrog steps L", range = 1:100, startvalue = NLEAP, format = "{:d}");
        tellheight = true, width = 480, halign = :left)
    sl_s, sl_eps, sl_L = sgS.sliders
    tlen = Label(left[8, 1], " ", halign = :left, tellwidth = false, color = :gray35, fontsize = 11)
    tq = left[9, 1] = GridLayout(halign = :left)
    tg_pre = Toggle(tq[1, 1], active = false)
    Label(tq[1, 2], "precondition both with Σ\n(C = Σ,  M = Σ⁻¹)", halign = :left, justification = :left, fontsize = 12)
    tg_jit = Toggle(tq[1, 3], active = false)
    Label(tq[1, 4], "jitter ε\n(HMC)", halign = :left, justification = :left, fontsize = 12)
    colgap!(tq, 6); colgap!(tq, 2, 18)

    # --- 3. run
    Label(left[10, 1], "3 · run both", font = :bold, halign = :left, tellwidth = false)
    sgR = SliderGrid(left[11, 1],
        (label = "iterations per run", range = BATCHES, startvalue = 100, format = "{:d}"),
        (label = "burn-in", range = 0:10:5000, startvalue = 0, format = "discard first {:d}");
        tellheight = true, width = 480, halign = :left)
    sl_batch, sl_burn = sgR.sliders
    bR = left[12, 1] = GridLayout(halign = :left)
    btn_run   = Button(bR[1, 1], label = "Run both", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 150, height = 28)
    btn_reset = Button(bR[1, 2], label = "Reset", height = 28)
    tg_eq     = Toggle(bR[1, 3], active = false)
    Label(bR[1, 4], "equal cost: Metropolis gets\nL + 1 iterations per HMC iteration", halign = :left,
          justification = :left, fontsize = 11)
    colgap!(bR, 2, 16); colgap!(bR, 3, 6)
    status = Label(left[13, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 34)
    rowgap!(left, 4)
    for sg in (sgD, sgS, sgR)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    P0 = Point2f[(0, 0)]; PN = Point2f[(NaN, NaN)]
    o_dat  = Observable(P0); o_err = Observable(Point2f[(0, 0), (0, 0)])
    o_true = Observable(Point2f[(0, 0), (1, 0)])
    o_fit  = [Observable(Point2f[(0, 0), (1, 0)]) for _ in 1:2]
    o_gx = Observable(collect(range(0, 1, length = NG))); o_gy = Observable(collect(range(0, 1, length = NG)))
    o_gz = Observable(zeros(NG, NG)); o_gl = Observable(zeros(NG, NG))
    o_lev1 = Observable(collect(1.0:3)); o_lev2 = Observable(collect(1.0:10))
    o_xell = Observable(PN); o_true2 = Observable(P0)
    o_vis  = [Observable(P0) for _ in 1:2]; o_path = [Observable(PN) for _ in 1:2]
    o_cur  = [Observable(P0) for _ in 1:2]; o_traj = Observable(PN)
    o_tr   = [Observable(P0) for _ in 1:2]; o_km = Observable(0.0); o_kband = Observable(Rect2f(0, 0, 1, 1))
    o_burn = Observable(Rect2f(0, 0, 0, 1)); vis_burn = Observable(true)
    o_err_ = [Observable(Point2f[(1, 1)]) for _ in 1:2]; o_ref = Observable(Point2f[(1, 1)]); vis_ref = Observable(true)
    o_kt = Observable(K_TRUE)

    # data panel
    lines!(axD, o_true; color = C_TRUE, linewidth = 2, linestyle = :dash, label = "true line")
    lines!(axD, o_fit[1]; color = C_MH, linewidth = 2.5, label = "Metropolis: current fit")
    lines!(axD, o_fit[2]; color = C_HMC, linewidth = 2.5, label = "HMC: current fit")
    linesegments!(axD, o_err; color = (C_DATA, 0.6), linewidth = 1.2)
    scatter!(axD, o_dat; color = C_DATA, markersize = 8, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 10, rowgap = -2, colgap = 6, nbanks = 2, patchsize = (14, 8),
               padding = (5, 5, 3, 3), backgroundcolor = (:white, 0.85))

    # maps
    for j in 1:2
        ax = axM[j]; c = (C_MH, C_HMC)[j]
        heatmap!(ax, o_gx, o_gy, o_gl; colormap = :Greys, alpha = 0.35)
        contour!(ax, o_gx, o_gy, o_gz; levels = o_lev2, color = (:gray40, 0.4), linewidth = 0.6)
        contour!(ax, o_gx, o_gy, o_gz; levels = o_lev1, color = C_MIS, linewidth = 1.5)
        lines!(ax, o_xell; color = C_POST, linewidth = 1.5, linestyle = :dash)
        scatter!(ax, o_vis[j]; color = (c, 0.3), markersize = 4)
        lines!(ax, o_path[j]; color = (c, 0.85), linewidth = 1.2, visible = tg_path.active)
        j == 2 && scatterlines!(ax, o_traj; color = (:black, 0.7), markersize = 3, linewidth = 1, visible = tg_path.active)
        scatter!(ax, o_true2; color = :white, marker = :star5, markersize = 16, strokecolor = :black, strokewidth = 1.2)
        scatter!(ax, o_cur[j]; color = c, markersize = 14, strokecolor = :black, strokewidth = 1.5)
    end

    # trace of k
    poly!(axT, o_kband; color = (C_POST, 0.08))
    poly!(axT, o_burn; color = (:gray50, 0.18), visible = vis_burn)
    hlines!(axT, o_km; color = C_POST, linestyle = :dash, linewidth = 1.2)
    lines!(axT, o_tr[1]; color = (C_MH, 0.9), linewidth = 1)
    lines!(axT, o_tr[2]; color = (C_HMC, 0.9), linewidth = 1)

    # convergence
    lines!(axE, o_ref; color = :gray30, linestyle = :dash, linewidth = 1.5, visible = vis_ref,
           label = "independent samples √(2/n)")
    lines!(axE, o_err_[1]; color = C_MH, linewidth = 2, label = "Metropolis")
    lines!(axE, o_err_[2]; color = C_HMC, linewidth = 2, label = "HMC")
    axislegend(axE; position = :lb, labelsize = 10, rowgap = -2, padding = (4, 4, 2, 2), patchsize = (14, 8),
               backgroundcolor = (:white, 0.85))

    # ------------------------------------------------------------ functions
    post() = exact_posterior(S[], P)
    "Proposal covariance (Metropolis) and mass matrix (HMC)."
    function precond()
        _, C = post()
        tg_pre.active[] ? (C, inv2(C)) : (I2, I2)
    end
    cost_mode() = tg_cost.active[]

    function new_noise!()
        rng_d[] = Xoshiro(rand(UInt32))
        xall[]  = XLIM[1] .+ (XLIM[2] - XLIM[1]) .* rand(rng_d[], NMAX)
        eall[]  = randn(rng_d[], NMAX)
    end

    function init_data!()
        kt, at, σ, n = sl_kt.value[], sl_at.value[], sl_sig.value[], sl_n.value[]
        used[] = (kt, at, σ, n)
        xd[] = xall[][1:n]
        yd[] = at .+ kt .* xd[] .+ σ .* eall[][1:n]
        S[]  = DataSums(xd[], yd[], σ)
        o_kt[] = kt
        o_true2[] = [Point2f(kt, at)]
        o_dat[]  = Point2f.(xd[], yd[])
        o_err[]  = vec(permutedims(hcat(Point2f.(xd[], yd[] .- σ), Point2f.(xd[], yd[] .+ σ))))
        o_true[] = [Point2f(XLIM[1], at + kt * XLIM[1]), Point2f(XLIM[2], at + kt * XLIM[2])]
        axD.title = @sprintf("data: N = %d,  k_true = %.2f,  a_true = %.1f,  σ = %.2g", n, kt, at, σ)
        lo, hi = extrema(vcat(yd[] .- σ, yd[] .+ σ, 0.0))
        limits!(axD, XLIM[1] - 0.3, XLIM[2] + 0.3, lo - 0.08(hi - lo), hi + 0.08(hi - lo))
        m, C = post()
        o_xell[] = vcat(ellipse(m, C, 1.0), ellipse(m, C, 2.0))
        o_km[] = m[1]
    end

    function window()
        tg_zoom.active[] || return (KLIM..., ALIM...)
        m, C = post(); sk, sa = sd2(C)
        return (m[1] - 4.5sk, m[1] + 4.5sk, m[2] - 4.5sa, m[2] + 4.5sa)
    end

    function draw_map!()
        k1, k2, a1, a2 = window()
        gx = collect(range(k1, k2, length = NG)); gy = collect(range(a1, a2, length = NG))
        Sd = S[]
        gz = [chi2(k, a, Sd) for k in gx, a in gy]
        cmin = chi2(lsq(Sd)[1]..., Sd)
        o_gx[] = gx; o_gy[] = gy
        o_lev1[] = cmin .+ DCHI
        o_lev2[] = cmin .+ 25 .* 2.0 .^ (0:14)
        o_gl[] = log10.(gz .- cmin .+ 1)
        o_gz[] = gz
        for ax in axM
            limits!(ax, k1, k2, a1, a2)
        end
    end

    function reset_chains!()
        θ0 = start[]                                  # (the random streams continue: every run is a new realisation)
        lp = logpost(θ0, S[], P)
        ch[1] = Chain(θ0, lp); ch[2] = Chain(θ0, lp)
        startlabel.text[] = @sprintf("start (k₀, a₀) = (%.2f, %.2f) – click in a map to change it.   ", θ0...) *
                            @sprintf("Priors: k ~ N(%.3g, %.3g²), a ~ N(%.3g, %.3g²)", PRIOR...)
    end

    "n HMC iterations and n (or n·(L+1) with 'equal cost') Metropolis iterations."
    function run_both!(n)
        Sd = S[]; C, M = precond(); Minv = inv2(M)
        s = sl_s.value[]; ε0 = sl_eps.value[]; L = sl_L.value[]
        nm = tg_eq.active[] ? n * (L + 1) : n
        c = ch[1]; θ = (c.ks[end], c.as[end])
        for _ in 1:nm
            θ, c.lp, acc = mh_step(rng_m[], θ, c.lp, s, C, Sd, P)
            c.nacc += acc
            push!(c.ks, θ[1]); push!(c.as, θ[2]); push!(c.cost, c.cost[end] + 1)
        end
        c = ch[2]; θ = (c.ks[end], c.as[end])
        for _ in 1:n
            ε = tg_jit.active[] ? ε0 * (0.8 + 0.4rand(rng_h[])) : ε0
            θ, acc, div, traj = hmc_step(rng_h[], θ, ε, L, M, Minv, Sd, P)
            c.nacc += acc; c.ndiv += div; c.traj = traj
            push!(c.ks, θ[1]); push!(c.as, θ[2]); push!(c.cost, c.cost[end] + L + 1)
        end
    end

    "Trajectory length in oscillation periods and the leapfrog stability limit."
    function draw_tlen!()
        _, C = post(); _, M = precond()
        λ1, λ2 = eig_prod(inv2(M), inv2(C))
        ω1, ω2 = sqrt(λ1), sqrt(λ2)
        ε = sl_eps.value[]; L = sl_L.value[]
        stable = ε * ω2 < 2
        f = stable ? [L * 2asin(ε * ω / 2) / 2π for ω in (ω1, ω2)] : [NaN, NaN]
        res = stable && !tg_jit.active[] && any(x -> x > 0.3 && abs(x - round(2x) / 2) < 0.06, f)
        tlen.text[] = stable ?
            @sprintf("HMC trajectory = %.2f, %.2f periods", f...) *
            (res ? "  –  RESONANCE (≈ multiple of ½): try jitter ε" : "  (¼ ideal, ½ reflection, 1 U-turn)") *
            @sprintf("  ·  stable for ε < %.3g", 2 / ω2) :
            @sprintf("HMC UNSTABLE: ε > 2/ω_max = %.3g  –  all trajectories diverge", 2 / ω2)
        tlen.color = !stable ? :red3 : res ? :darkorange3 : :gray35
    end

    "Index (iteration) at which a chain first enters the 99.7 % ellipse of the posterior; 0 if never."
    function reached(c::Chain)
        m, C = post(); Λ = inv2(C)
        i = findfirst(j -> quad(Λ, (c.ks[j] - m[1], c.as[j] - m[2])) < DCHI[3], eachindex(c.ks))
        return i === nothing ? 0 : i - 1
    end

    function draw_chains!()
        m, C = post(); Λ = inv2(C); sk, sa = sd2(C)
        nb = sl_burn.value[]
        cm = cost_mode()
        line(k, a) = [Point2f(XLIM[1], a + k * XLIM[1]), Point2f(XLIM[2], a + k * XLIM[2])]
        xmax = 10.0; emin = Inf; emax = 0.0
        for j in 1:2
            c = ch[j]; n = length(c.ks) - 1
            o_fit[j][] = line(c.ks[end], c.as[end])
            o_vis[j][]  = Point2f.(c.ks, c.as)
            o_path[j][] = Point2f.(c.ks[max(1, end - NPATH):end], c.as[max(1, end - NPATH):end])
            o_cur[j][]  = [Point2f(c.ks[end], c.as[end])]
            xs = cm ? c.cost : collect(0.0:n)
            o_tr[j][] = Point2f.(xs, c.ks)
            xmax = max(xmax, xs[end])
            # running mean error after the burn-in
            if n > nb
                kk = @view c.ks[nb+2:end]; aa = @view c.as[nb+2:end]
                sk_ = cumsum(kk); sa_ = cumsum(aa); N = length(kk)
                idx = logidx(N, NPLOT)
                d = [sqrt(quad(Λ, (sk_[i] / i - m[1], sa_[i] / i - m[2]))) for i in idx]
                d = max.(d, 1e-4)
                x = cm ? c.cost[nb .+ 1 .+ idx] : Float64.(nb .+ idx)
                o_err_[j][] = Point2f.(x, d)
                emin = min(emin, minimum(d)); emax = max(emax, maximum(d))
            else
                o_err_[j][] = Point2f[(NaN, NaN)]
            end
            # table
            nacc = c.nacc
            ri = reached(c)
            vals = Any["$n", @sprintf("%.0f", c.cost[end]),
                       n == 0 ? "–" : @sprintf("%.0f %%", 100nacc / n) * (j == 2 && c.ndiv > 0 ? " ($(c.ndiv) div.)" : ""),
                       ri == 0 && n > 0 && quad(Λ, (c.ks[1] - m[1], c.as[1] - m[2])) >= DCHI[3] ? "not yet" :
                       @sprintf("it. %d (%.0f ev.)", ri, c.cost[ri+1])]
            if n > nb + 1
                kk = c.ks[nb+2:end]; aa = c.as[nb+2:end]
                ek, ea = ess(kk), ess(aa)
                ev = c.cost[end] - c.cost[nb+1]
                push!(vals, fmt(mean_(kk)) * ",  " * fmt(mean_(aa)),
                      fmt(sqrt(var_(kk))) * ",  " * fmt(sqrt(var_(aa))),
                      isnan(ek) ? "(≥ 50 samples)" : @sprintf("%.0f,  %.0f", ek, ea),
                      isnan(ek) ? "–" : @sprintf("%.3g", 1000 * min(ek, ea) / ev))
            else
                append!(vals, ["–", "–", "–", "–"])
            end
            for i in eachindex(vals)
                cells[i, j].text[] = vals[i]
            end
        end
        exactlab.text[] = @sprintf("exact posterior:  k = %.3g ± %.3g,   a = %.3g ± %.3g,   ρ = %.2f", m[1], sk, m[2], sa, corr2(C))
        # HMC: last trajectory
        o_traj[] = isempty(ch[2].traj) ? PN : filter(p -> all(isfinite, p), ch[2].traj)
        # trace and convergence limits
        k1, k2, _, _ = window()
        limits!(axT, cm ? -0.01xmax : -0.02xmax, 1.02xmax, k1, k2)
        axT.xlabel = cm ? "model evaluations" : "iteration"
        o_kband[] = Rect2f(cm ? -0.01xmax : -0.02xmax, m[1] - sk, 1.05xmax, 2sk)
        vis_burn[] = !cm
        o_burn[] = Rect2f(0, k1, min(nb, max(length(ch[1].ks), length(ch[2].ks)) - 1), k2 - k1)
        axE.xlabel = cm ? "model evaluations" : "iteration"
        vis_ref[] = !cm
        nmax = maximum(length(c.ks) - 1 for c in ch)
        if nmax > nb + 1
            nr = collect(1:max(nmax - nb, 1)); idx = logidx(length(nr), 400)
            o_ref[] = Point2f.(nb .+ nr[idx], sqrt.(2 ./ nr[idx]))
            isfinite(emin) || (emin = 0.01; emax = 1.0)
            emin = min(emin, sqrt(2 / (nmax - nb))); emax = max(emax, sqrt(2.0))
            xlo = max(nb + 1, 1); xhi = max(cm ? xmax : nmax, xlo + 1)
            limits!(axE, xlo, xhi * 1.1, emin / 1.5, emax * 1.5)
        else
            o_ref[] = Point2f[(1, 1)]
            limits!(axE, 1, 10, 0.01, 10)
        end
    end

    data_pending() = (sl_kt.value[], sl_at.value[], sl_sig.value[], sl_n.value[]) != used[]
    function set_status!(msg; warn = false)
        status.text[] = msg
        status.color = warn ? :red3 : :gray20
    end

    function initialise!()
        init_data!(); reset_chains!(); draw_map!(); draw_tlen!(); draw_chains!()
        set_status!("Data initialised. Press \"Run both\" – both samplers start at the same point.")
    end
    function restart!(why)
        reset_chains!(); draw_tlen!(); draw_chains!()
        set_status!(why * @sprintf(" → both chains restarted at (k₀, a₀) = (%.2f, %.2f).", start[]...))
    end

    # ------------------------------------------------------------ callbacks
    on(_ -> initialise!(), btn_init.clicks)
    on(btn_noise.clicks) do _
        new_noise!(); initialise!()
        set_status!("New noise realisation (new x positions and ε). Data and χ²(k, a) recomputed.")
    end
    for sl in (sl_kt, sl_at, sl_sig, sl_n)
        on(sl.value) do _
            data_pending() ? set_status!("Data settings changed – press \"Initialise data\" to apply them.", warn = true) :
                             set_status!(" ")
        end
    end
    on(_ -> restart!("reset"), btn_reset.clicks)
    on(btn_run.clicks) do _
        run_both!(sl_batch.value[]); draw_chains!()
        data_pending() && set_status!("Note: the data sliders were changed but not applied (press \"Initialise data\").",
                                      warn = true)
    end
    on(tg_pre.active) do p
        draw_tlen!()
        restart!(p ? "preconditioning ON (Metropolis: proposal s²Σ, try s ≈ 1.5; HMC: M = Σ⁻¹, try ε ≈ 0.3, L ≈ 5)" :
                     "preconditioning OFF (C = I, M = I)")
    end
    for sl in (sl_s, sl_eps, sl_L)
        on(_ -> draw_tlen!(), sl.value)
    end
    on(_ -> draw_tlen!(), tg_jit.active)
    on(_ -> draw_chains!(), sl_burn.value)
    on(_ -> draw_chains!(), tg_cost.active)
    on(_ -> (draw_map!(); draw_chains!()), tg_zoom.active)
    for ax in axM
        register_interaction!(ax, :pick_start) do event::MouseEvent, _
            if event.type === MouseEventTypes.leftclick
                start[] = (clamp(event.data[1], KLIM...), clamp(event.data[2], ALIM...))
                restart!("starting point changed")
            end
            return Consume(false)
        end
    end

    # ------------------------------------------------------- initialisation
    rng_d[] = Xoshiro(SEED)
    xall[]  = XLIM[1] .+ (XLIM[2] - XLIM[1]) .* rand(rng_d[], NMAX)
    eall[]  = randn(rng_d[], NMAX)
    initialise!()
    return fig, (init = initialise!, run = n -> (run_both!(n); draw_chains!()), reset = () -> restart!("reset"),
                 setstart = θ -> (start[] = θ; restart!("starting point changed")),
                 sliders = (kt = sl_kt, at = sl_at, sig = sl_sig, n = sl_n, s = sl_s, eps = sl_eps, L = sl_L,
                            batch = sl_batch, burn = sl_burn),
                 toggles = (cost = tg_cost, zoom = tg_zoom, path = tg_path, pre = tg_pre, jit = tg_jit, eq = tg_eq),
                 chains = ch, cells = cells)
end

# =============================================================================
# RUN
# =============================================================================
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    fig, app = build_app()
    screen = display(fig)
    isinteractive() || wait(screen)     # keep the window open when run from a terminal
end
