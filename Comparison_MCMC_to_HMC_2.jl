#=
Comparison_MCMC_to_HMC_2 – Metropolis versus HMC when the number of parameters grows
=====================================================================================

Model: a polynomial with D = 1 … 10 coefficients on −1 ≤ x ≤ 1,
      y = Σⱼ cⱼ φⱼ(x),   j = 0 … D−1,
with φⱼ either the Legendre polynomials Pⱼ(x) (nearly orthogonal for points spread
over [−1, 1]: the coefficients are almost uncorrelated) or the monomials xʲ (strongly
correlated coefficients). The model is linear in the coefficients, so with Gaussian
noise and the prior cⱼ ~ N(0, 2²) the posterior is an exact Gaussian:
      U(c) = −ln p(c|d) = ½ cᵀΛc − hᵀc + const,   Λ = ΦᵀΦ/σ² + I/σₚ²,  h = Φᵀy/σ²,
      mean m = Λ⁻¹h,  covariance Σ = Λ⁻¹,  gradient ∇U = Λc − h.

Samplers (both start at the same point, the default is c = 0):
  Metropolis   c′ ~ N(c, s²C)                      1 model evaluation per iteration
  HMC          p ~ N(0, C⁻¹), L leapfrog steps of ε  L + 1 gradient evaluations per iteration
  with C = Σ ("precondition", as if learned in a pilot run) or C = I.

"Optimal tuning" (on by default) sets the parameters from the theory for a Gaussian
target with the frequencies ω (ω² = eigenvalues of CΛ; all ω = 1 when preconditioned):
  Metropolis   s = 2.38/√D / ω_max                 (acceptance ≈ 23 – 44 %)
  HMC          ε = 0.8·D^(−1/4) / ω_max,  L = round(π/2 / (ω_min ε))   (a quarter period)

The SCAN runs both samplers for D = 1 … 10 with the same budget of model evaluations,
30 times each, starting from exact posterior draws, and plots the TRUE effective sample
size per 1000 evaluations against D (from the error of the sample means; the batch-means
estimate shown in the table is far too optimistic for chains that mix badly – for the
monomial basis without preconditioning by a factor of up to 200).

What the scan shows (30 000 evaluations per run; values vary by ±30 % between scans):
                                       Metropolis          HMC          HMC / Metropolis
                                       D = 1   D = 10      D = 1   D = 10   at D = 10
  preconditioned (any basis)           ≈ 210   ≈ 29        ≈ 260   120 – 220     4 – 7
  Legendre, C = I                      ≈ 300   ≈ 7         ≈ 550   50 – 65        7 – 10
  monomials, C = I (ω_max/ω_min ≈ 70)  ≈ 250   ≈ 0.1       ≈ 350   ≈ 3           20 – 30
  * Metropolis loses a factor ≈ 1/D even in the best case (theory: its step must shrink
    as D^(−1/2) and it needs O(D) iterations per independent sample); HMC only ≈ D^(−1/4)
    (its leapfrog step must shrink only as D^(−1/4)). At D = 1 the two are similar; the
    ratio grows with D. Extrapolated: ≈ 15× at D = 100, ≈ 100× at D = 1000.
  * A badly conditioned posterior (monomials, no preconditioning) hurts both, Metropolis
    far more: its step is set by the narrowest direction and it then crawls along the
    widest one. Preconditioning (C = Σ) makes both samplers independent of the
    parametrisation – Legendre and monomials then give the same efficiency.
  * HMC's values jump somewhat from D to D: L is an integer, so the trajectory length
    varies between ≈ ¼ and ⅓ of a period.
  Hollow markers: ESS per run below 3 – the chain hardly moved in that run.

Run from a terminal:      julia Comparison_MCMC_to_HMC_2.jl
or from the Julia REPL:   include("Comparison_MCMC_to_HMC_2.jl")
=#

using Pkg
Pkg.activate(@__DIR__)

using GLMakie
using LinearAlgebra
using Printf
using Random

# =============================================================================
# USER INPUT
# =============================================================================
const DMAX     = 10                 # largest number of coefficients
const CTRUE    = [0.5, 1.0, -0.6, 0.4, 0.3, -0.25, 0.2, -0.15, 0.1, 0.05]   # true coefficients c₀ … c₉
const D_START  = 5                  # default number of coefficients
const SIGMA    = 0.2                # default noise standard deviation
const NDATA    = 40                 # default number of data points
const NMAX     = 200                # largest number of data points
const SIGMA_P  = 2.0                # prior standard deviation of every coefficient (mean 0)
const SEED     = 1
const BATCHES  = [1, 10, 100, 1000] # HMC iterations per press of "Run"
const BUDGETS  = [3_000, 10_000, 30_000, 100_000]   # model evaluations per sampler, D and run in the scan
const NREP     = 30                 # independent runs per D in the scan (true ESS from their error)
const NPATH    = 100                # number of last steps drawn as a path
const NSPAG    = 20                 # number of last states drawn in the data panel
const NPLOT    = 1500               # points of the (log-spaced) convergence curves
const DIVERGE  = 1000.0             # ΔH above this: divergent trajectory
const ESS_FLOOR = 3                 # ESS per run below this: the chain hardly moved (hollow marker; true value may be lower)

const C_MH    = :darkorange2        # Metropolis
const C_HMC   = :dodgerblue3        # HMC
const C_DATA  = :black
const C_TRUE  = :gray40
const C_POST  = :black

# =============================================================================
# MATHEMATICS (independent of the plotting)
# =============================================================================
"Design matrix Φ (N × D): Legendre polynomials P₀ … P_{D−1} or monomials x⁰ … x^{D−1}."
function basis(x, D, kind::Symbol)
    Φ = zeros(length(x), D)
    for (i, xi) in enumerate(x)
        if kind === :legendre
            p0, p1 = 1.0, xi
            Φ[i, 1] = 1.0
            D > 1 && (Φ[i, 2] = xi)
            for n in 2:D-1                          # (n) Pₙ = (2n−1) x Pₙ₋₁ − (n−1) Pₙ₋₂
                p0, p1 = p1, ((2n - 1) * xi * p1 - (n - 1) * p0) / n
                Φ[i, n+1] = p1
            end
        else
            for j in 1:D
                Φ[i, j] = xi^(j - 1)
            end
        end
    end
    return Φ
end

"The inverse problem: data, precision Λ, h, exact mean m and covariance Σ."
struct Problem
    D::Int
    kind::Symbol
    x::Vector{Float64}
    y::Vector{Float64}
    σ::Float64
    Φ::Matrix{Float64}
    Λ::Matrix{Float64}
    h::Vector{Float64}
    m::Vector{Float64}
    Σ::Matrix{Float64}
end
function Problem(D, kind, x, noise, σ)
    Φ = basis(x, D, kind)
    y = Φ * CTRUE[1:D] .+ σ .* noise
    Λ = Symmetric(Φ' * Φ ./ σ^2 + I / SIGMA_P^2)
    h = Φ' * y ./ σ^2
    Σ = Symmetric(inv(Λ))
    return Problem(D, kind, collect(x), y, σ, Φ, Matrix(Λ), h, Λ \ h, Matrix(Σ))
end
U(c, P::Problem)  = 0.5 * dot(c, P.Λ * c) - dot(P.h, c)
dU(c, P::Problem) = P.Λ * c .- P.h
chi2(c, P::Problem) = sum(abs2, P.y .- P.Φ * c) / P.σ^2
"Squared Mahalanobis distance from the posterior mean."
maha2(c, P::Problem) = (δ = c .- P.m; dot(δ, P.Λ * δ))
"99.7 % quantile of χ² with D degrees of freedom (Wilson–Hilferty)."
q997(D) = D * (1 - 2 / (9D) + 2.748 * sqrt(2 / (9D)))^3

"Sampler settings: proposal covariance C (HMC: M = C⁻¹), their Cholesky factors, s, ε, L, frequencies."
struct Tune
    C::Matrix{Float64}
    Lc::LowerTriangular{Float64,Matrix{Float64}}
    LM::LowerTriangular{Float64,Matrix{Float64}}
    s::Float64
    ε::Float64
    L::Int
    ωmin::Float64
    ωmax::Float64
end
"Frequencies ω (ω² = eigenvalues of CΛ) of the particle in the Gaussian bowl."
function freqs(P::Problem, C)
    R = sqrt(Symmetric(C))
    ω = sqrt.(max.(eigvals(Symmetric(R * P.Λ * R)), 0.0))
    return minimum(ω), maximum(ω)
end
function Tune(P::Problem, pre::Bool, s, ε, L)
    C = pre ? P.Σ : Matrix(1.0I, P.D, P.D)
    ωmin, ωmax = freqs(P, C)
    return Tune(C, cholesky(Symmetric(C)).L, cholesky(Symmetric(inv(Symmetric(C)))).L, s, ε, L, ωmin, ωmax)
end
"Optimal settings for a Gaussian target (see the header): (s, ε, L)."
function optimal(P::Problem, pre::Bool)
    C = pre ? P.Σ : Matrix(1.0I, P.D, P.D)
    ωmin, ωmax = freqs(P, C)
    s = 2.38 / sqrt(P.D) / ωmax
    ε = 0.8 * P.D^(-0.25) / ωmax
    L = clamp(round(Int, (π / 2) / (ωmin * ε)), 1, 500)
    return s, ε, L
end

"One Metropolis step. Returns (c, U(c), accepted)."
function mh_step(rng, c, u, T::Tune, P::Problem)
    cp = c .+ T.s .* (T.Lc * randn(rng, P.D))
    up = U(cp, P)
    return log(rand(rng)) < u - up ? (cp, up, true) : (c, u, false)
end
"One HMC iteration with step ε. Returns (c, accepted, divergent, trajectory)."
function hmc_step(rng, c, ε, T::Tune, P::Problem; keep = false)
    p  = T.LM * randn(rng, P.D)
    H0 = U(c, P) + 0.5 * dot(p, T.C * p)
    t  = copy(c)
    ph = p .- 0.5ε .* dU(t, P)
    traj = keep ? [copy(t)] : Vector{Float64}[]
    for l in 1:T.L
        t .+= ε .* (T.C * ph)
        g = dU(t, P)
        keep && push!(traj, copy(t))
        (!all(isfinite, t) || maximum(abs, t) > 1e8) && return c, false, true, traj
        ph .-= (l < T.L ? ε : 0.5ε) .* g
    end
    ΔH = U(t, P) + 0.5 * dot(ph, T.C * ph) - H0
    div = !(ΔH < DIVERGE)
    acc = !div && log(rand(rng)) < -ΔH
    return acc ? t : c, acc, div, traj
end

mean_(v) = sum(v) / length(v)
var_(v) = (m = mean_(v); sum(abs2, v .- m) / (length(v) - 1))
"Effective sample size by batch means (capped at n)."
function ess(v)
    n = length(v)
    n < 50 && return NaN
    b = floor(Int, sqrt(n)); nb = n ÷ b
    bm = [mean_(view(v, (i-1)*b+1:i*b)) for i in 1:nb]
    vb = var_(bm); vx = var_(v)
    (vb <= 0 || vx <= 0) && return NaN
    return min(n, n * vx / (b * vb))
end
"Smallest ESS over the coefficients of the samples (columns = iterations)."
essmin(S::AbstractMatrix) = minimum(ess(view(S, j, :)) for j in axes(S, 1))
logidx(N, n) = N <= n ? collect(1:N) : unique(round.(Int, exp.(range(0, log(N), length = n))))

"""
Scan: for D = 1 … DMAX run both samplers NREP times with the optimal settings and the
same budget of model evaluations. Every run starts at an EXACT draw from the posterior
(possible here because the posterior is known), so the chain is stationary from the
first iteration. The TRUE effective sample size follows from the error of the sample
means over the runs, measured in units of the posterior (Mahalanobis distance d from the
exact mean, the same measure as in the convergence plot):  1/ESS = mean over runs of d²/D.
For independent samples d²/D ≈ 1/n. (Unlike "the smallest ESS over the coefficients" this
measure is the same in every parametrisation – Legendre or monomials – and much less noisy.)
(The batch-means estimate used in the table is far too optimistic for badly mixing chains.)
Returns per sampler a DMAX × 4 matrix: ESS per 1000 evaluations, lower and upper end of
its ≈68 % uncertainty, ESS per run.
"""
function scan(kind, pre, x, noise, σ, budget, rng; jitter = true)
    res = Dict(:mh => fill(NaN, DMAX, 4), :hmc => fill(NaN, DMAX, 4))
    δ = sqrt(2 / (NREP - 1))                         # relative uncertainty of a variance from NREP values
    for D in 1:DMAX
        P = Problem(D, kind, x, noise, σ)
        s, ε, L = optimal(P, pre)
        T = Tune(P, pre, s, ε, L)
        Lp = cholesky(Symmetric(P.Σ)).L
        d2 = Dict(:mh => 0.0, :hmc => 0.0)
        nh = max(budget ÷ (L + 1), 1)
        for r in 1:NREP
            c0 = P.m .+ Lp * randn(rng, D)           # exact posterior draw
            c = copy(c0); u = U(c, P); acc = zeros(D)
            for _ in 1:budget
                c, u, _ = mh_step(rng, c, u, T, P); acc .+= c
            end
            d2[:mh] += maha2(acc ./ budget, P) / D
            c = copy(c0); acc = zeros(D)
            for _ in 1:nh
                εi = jitter ? ε * (0.8 + 0.4rand(rng)) : ε
                c, _, _, _ = hmc_step(rng, c, εi, T, P); acc .+= c
            end
            d2[:hmc] += maha2(acc ./ nh, P) / D
        end
        for (k, cost) in ((:mh, budget), (:hmc, nh * (L + 1)))
            e = NREP / d2[k]                           # ESS per run
            res[k][D, :] = [1000e / cost, 1000e / cost / (1 + δ), 1000e / cost / max(1 - δ, 0.2), e]
        end
    end
    return res
end

fmt(v) = isnan(v) ? "–" : @sprintf("%.3g", abs(v) < 1e-9 ? 0.0 : v)

# =============================================================================
# APPLICATION
# =============================================================================
mutable struct Chain
    cs::Vector{Vector{Float64}}     # visited states (cs[1] = start)
    cost::Vector{Float64}           # model evaluations up to each iteration
    nacc::Int
    ndiv::Int
    u::Float64                      # U at the current state (Metropolis)
    traj::Vector{Vector{Float64}}   # last HMC trajectory
end
Chain(c, P) = Chain([copy(c)], [0.0], 0, 0, U(c, P), Vector{Float64}[])

function build_app()
    # ---------------------------------------------------------------- state
    rng_d = Ref(Xoshiro(SEED)); rng_m = Xoshiro(SEED + 1000); rng_h = Xoshiro(SEED + 2000); rng_s = Xoshiro(SEED + 3000)
    xall  = Ref(zeros(NMAX)); eall = Ref(zeros(NMAX))
    prob  = Ref{Problem}()
    tune  = Ref{Tune}()
    ch    = Chain[]
    used  = Ref((D_START, :legendre, SIGMA, NDATA))
    syncing = Ref(false)
    scanres = Ref{Any}(nothing)

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Comparison 2: Metropolis versus HMC when the number of parameters grows  (polynomial fit, D ≤ 10)",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(520))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")

    top = right[1, 1] = GridLayout()
    axP = Axis(top[1, 1]; title = "chains projected on two coefficients")
    axS = Axis(top[1, 2]; xlabel = "number of parameters D", ylabel = "ESS per 1000 model evaluations",
               yscale = log10, xticks = 1:DMAX, title = "scan: efficiency versus D (press \"Scan\")")
    colsize!(top, 1, Relative(0.45)); colgap!(top, 18)
    axT = Axis(right[2, 1]; ylabel = "coefficient")
    conv = right[3, 1] = GridLayout()
    axE = Axis(conv[1, 1]; xscale = log10, yscale = log10, ylabel = "error of the mean / √D",
               title = "convergence of the running mean (after burn-in)")
    tab = conv[1, 2] = GridLayout(valign = :center, tellheight = false)
    colgap!(conv, 20)
    rowsize!(right, 1, Auto(1.5)); rowsize!(right, 2, Auto(0.6)); rowsize!(right, 3, Auto(1.0)); rowgap!(right, 6)
    for a in (axD, axP, axS, axT, axE), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
    end

    rows = ["iterations", "model evaluations", "accepted", "reached posterior at", "largest error of a mean",
            "ESS estimate (smallest)", "ESS per 1000 evaluations"]
    Label(tab[1, 2], "Metropolis", color = C_MH, font = :bold, fontsize = 13)
    Label(tab[1, 3], "HMC", color = C_HMC, font = :bold, fontsize = 13)
    cells = Matrix{Label}(undef, length(rows), 2)
    for (i, r) in enumerate(rows)
        Label(tab[i+1, 1], r, halign = :right, fontsize = 12, color = :gray25)
        for j in 1:2
            cells[i, j] = Label(tab[i+1, j+1], "–", fontsize = 12, width = 120)
        end
    end
    tunelab = Label(tab[length(rows)+2, 1:3], " ", fontsize = 11, color = :gray30, justification = :center)
    rowgap!(tab, 1); colgap!(tab, 10)

    tg = right[4, 1] = GridLayout(halign = :center, tellwidth = false)
    tg_cost = Toggle(tg[1, 1], active = true);  Label(tg[1, 2], "x-axis: model evaluations", halign = :left)
    tg_path = Toggle(tg[1, 3], active = true);  Label(tg[1, 4], "paths / last trajectory", halign = :left)
    tg_spag = Toggle(tg[1, 5], active = true);  Label(tg[1, 6], "last fits in the data panel", halign = :left)
    colgap!(tg, 6)
    for c in (2, 4)
        colgap!(tg, c, 18)
    end

    # --- 1. model and data
    Label(left[2, 1], "1 · model  y = Σ cⱼ φⱼ(x),  j = 0 … D−1,  priors cⱼ ~ N(0, 2²)", font = :bold, halign = :left,
          tellwidth = false)
    mg = left[3, 1] = GridLayout(halign = :left)
    Label(mg[1, 1], "basis φⱼ:", halign = :right)
    menu = Menu(mg[1, 2], options = ["Legendre Pⱼ(x)  (≈ uncorrelated)", "monomials xʲ  (correlated)"],
                default = "Legendre Pⱼ(x)  (≈ uncorrelated)", width = 300, height = 28)
    sgD = SliderGrid(left[4, 1],
        (label = "parameters D", range = 1:DMAX, startvalue = D_START, format = "{:d}"),
        (label = "noise σ", range = 0.02:0.02:1.0, startvalue = SIGMA, format = "{:.2f}"),
        (label = "N points", range = 5:NMAX, startvalue = NDATA, format = "{:d}"),
        (label = "show coefficient j", range = 0:DMAX-1, startvalue = 1, format = "{:d}");
        tellheight = true, width = 480, halign = :left)
    sl_D, sl_sig, sl_n, sl_j = sgD.sliders
    bD = left[5, 1] = GridLayout(halign = :left)
    btn_init  = Button(bD[1, 1], label = "Initialise data", buttoncolor = RGBf(0.80, 0.88, 1.0), height = 28)
    btn_noise = Button(bD[1, 2], label = "new noise realisation", height = 28)

    # --- 2. samplers
    nice = sort(unique(round.(vcat([m .* 10.0^e for e in -4:1 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3)))
    Label(left[6, 1], rich(rich("2 · Metropolis", color = C_MH), "  c′ ~ N(c, s²C)   and   ",
                           rich("HMC", color = C_HMC), "  (ε, L, M = C⁻¹)"), font = :bold, halign = :left, tellwidth = false)
    tq = left[7, 1] = GridLayout(halign = :left)
    tg_pre = Toggle(tq[1, 1], active = true)
    Label(tq[1, 2], "precondition both\n(C = Σ, else C = I)", halign = :left, justification = :left, fontsize = 12)
    tg_opt = Toggle(tq[1, 3], active = true)
    Label(tq[1, 4], "optimal tuning\n(sets s, ε, L)", halign = :left, justification = :left, fontsize = 12)
    tg_jit = Toggle(tq[1, 5], active = true)
    Label(tq[1, 6], "jitter ε", halign = :left, fontsize = 12)
    colgap!(tq, 6); colgap!(tq, 2, 14); colgap!(tq, 4, 14)
    sgS = SliderGrid(left[8, 1],
        (label = "Metropolis step s", range = filter(v -> 1e-4 <= v <= 5, nice), startvalue = 1.0, format = "{:.3g}"),
        (label = "HMC leapfrog step ε", range = filter(v -> 1e-4 <= v <= 3, nice), startvalue = 0.5, format = "{:.3g}"),
        (label = "HMC leapfrog steps L", range = 1:500, startvalue = 3, format = "{:d}");
        tellheight = true, width = 480, halign = :left)
    sl_s, sl_eps, sl_L = sgS.sliders

    # --- 3. run and scan
    Label(left[9, 1], "3 · run both  (start: all cⱼ = 0)   and   scan D = 1 … 10", font = :bold, halign = :left, tellwidth = false)
    sgR = SliderGrid(left[10, 1],
        (label = "HMC iterations per run", range = BATCHES, startvalue = 100, format = "{:d}"),
        (label = "burn-in", range = 0:10:5000, startvalue = 0, format = "discard first {:d}"),
        (label = "scan budget", range = BUDGETS, startvalue = 30_000, format = "{:d} evaluations");
        tellheight = true, width = 480, halign = :left)
    sl_batch, sl_burn, sl_budget = sgR.sliders
    bR = left[11, 1] = GridLayout(halign = :left)
    btn_run   = Button(bR[1, 1], label = "Run both", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 110, height = 28)
    btn_reset = Button(bR[1, 2], label = "Reset", height = 28)
    tg_eq     = Toggle(bR[1, 3], active = true)
    Label(bR[1, 4], "equal cost", halign = :left, fontsize = 12)
    btn_scan  = Button(bR[1, 5], label = "Scan D = 1 … 10", buttoncolor = RGBf(1.0, 0.90, 0.78), height = 28)
    colgap!(bR, 2, 14); colgap!(bR, 3, 4); colgap!(bR, 4, 14)
    status = Label(left[12, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 52)
    rowgap!(left, 4)
    for sg in (sgD, sgS, sgR)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    PN = Point2f[(NaN, NaN)]; P0 = Point2f[(0, 0)]
    xs = collect(range(-1, 1, length = 201))
    o_dat = Observable(P0); o_err = Observable(Point2f[(0, 0), (0, 0)]); o_true = Observable(P0)
    o_fit = [Observable(P0) for _ in 1:2]; o_spag = [Observable(PN) for _ in 1:2]
    o_ell = Observable(PN); o_mtrue = Observable(P0); o_mean = Observable(P0)
    o_vis = [Observable(P0) for _ in 1:2]; o_path = [Observable(PN) for _ in 1:2]; o_cur = [Observable(P0) for _ in 1:2]
    o_traj = Observable(PN)
    o_tr = [Observable(P0) for _ in 1:2]; o_band = Observable(Rect2f(0, 0, 1, 1)); o_bm = Observable(0.0); o_bt = Observable(0.0)
    o_burn = Observable(Rect2f(0, 0, 0, 1)); vis_burn = Observable(false)
    o_conv = [Observable(Point2f[(1, 1)]) for _ in 1:2]; o_ref = Observable(Point2f[(1, 1)]); vis_ref = Observable(false)
    o_sc   = [Observable(PN) for _ in 1:2]; o_scbad = [Observable(PN) for _ in 1:2]; o_scbar = [Observable(Point2f[(NaN, NaN), (NaN, NaN)]) for _ in 1:2]
    o_th   = [Observable(PN) for _ in 1:2]

    # data panel
    lines!(axD, o_true; color = C_TRUE, linewidth = 2, linestyle = :dash, label = "true polynomial")
    for j in 1:2
        lines!(axD, o_spag[j]; color = ((C_MH, C_HMC)[j], 0.25), linewidth = 1, visible = tg_spag.active)
    end
    lines!(axD, o_fit[1]; color = C_MH, linewidth = 2.5, label = "Metropolis: current fit")
    lines!(axD, o_fit[2]; color = C_HMC, linewidth = 2.5, label = "HMC: current fit")
    linesegments!(axD, o_err; color = (C_DATA, 0.5), linewidth = 1)
    scatter!(axD, o_dat; color = C_DATA, markersize = 6, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 10, rowgap = -2, colgap = 6, nbanks = 2, patchsize = (14, 8),
               padding = (5, 5, 3, 3), backgroundcolor = (:white, 0.85))

    # projection
    lines!(axP, o_ell; color = C_POST, linewidth = 1.5, linestyle = :dash, label = "exact posterior 1σ, 2σ")
    for j in 1:2
        c = (C_MH, C_HMC)[j]
        scatter!(axP, o_vis[j]; color = (c, 0.25), markersize = 4)
        lines!(axP, o_path[j]; color = (c, 0.8), linewidth = 1.1, visible = tg_path.active)
    end
    scatterlines!(axP, o_traj; color = (:black, 0.7), markersize = 3, linewidth = 1, visible = tg_path.active,
                  label = "last HMC trajectory")
    scatter!(axP, o_mtrue; color = :white, marker = :star5, markersize = 15, strokecolor = :black, strokewidth = 1.2,
             label = "true coefficients")
    scatter!(axP, o_cur[1]; color = C_MH, markersize = 13, strokecolor = :black, strokewidth = 1.5, label = "Metropolis")
    scatter!(axP, o_cur[2]; color = C_HMC, markersize = 13, strokecolor = :black, strokewidth = 1.5, label = "HMC")
    axislegend(axP; position = :rt, labelsize = 9, rowgap = -3, padding = (4, 4, 2, 2), patchsize = (12, 8),
               backgroundcolor = (:white, 0.85))

    # scan
    for j in 1:2
        c = (C_MH, C_HMC)[j]
        lines!(axS, o_th[j]; color = (c, 0.6), linestyle = :dash, linewidth = 1.5,
               label = j == 1 ? "∝ 1/D" : "∝ D^(−1/4)")
        linesegments!(axS, o_scbar[j]; color = c, linewidth = 1.5)
        scatterlines!(axS, o_sc[j]; color = c, markersize = 11, linewidth = 2, label = j == 1 ? "Metropolis" : "HMC")
        scatter!(axS, o_scbad[j]; color = :white, strokecolor = c, strokewidth = 2, markersize = 11)
    end
    axislegend(axS; position = :rt, labelsize = 10, rowgap = -2, padding = (4, 4, 2, 2), patchsize = (14, 8),
               backgroundcolor = (:white, 0.85))
    xlims!(axS, 0.5, DMAX + 0.5); ylims!(axS, 1, 1000)

    # trace
    poly!(axT, o_band; color = (C_POST, 0.08))
    poly!(axT, o_burn; color = (:gray50, 0.18), visible = vis_burn)
    hlines!(axT, o_bm; color = C_POST, linestyle = :dash, linewidth = 1.2)
    hlines!(axT, o_bt; color = C_TRUE, linestyle = :dot, linewidth = 1.2)
    lines!(axT, o_tr[1]; color = (C_MH, 0.9), linewidth = 1)
    lines!(axT, o_tr[2]; color = (C_HMC, 0.9), linewidth = 1)

    # convergence
    lines!(axE, o_ref; color = :gray30, linestyle = :dash, linewidth = 1.5, visible = vis_ref,
           label = "independent samples 1/√n")
    lines!(axE, o_conv[1]; color = C_MH, linewidth = 2, label = "Metropolis")
    lines!(axE, o_conv[2]; color = C_HMC, linewidth = 2, label = "HMC")
    axislegend(axE; position = :lb, labelsize = 10, rowgap = -2, padding = (4, 4, 2, 2), patchsize = (14, 8),
               backgroundcolor = (:white, 0.85))

    # ------------------------------------------------------------ functions
    kind() = menu.i_selected[] == 2 ? :monomial : :legendre
    function set_status!(msg; warn = false)
        status.text[] = msg
        status.color = warn ? :red3 : :gray20
    end
    function new_noise!()
        rng_d[] = Xoshiro(rand(UInt32))
        xall[]  = sort(2 .* rand(rng_d[], NMAX) .- 1)
        eall[]  = randn(rng_d[], NMAX)
    end
    "The N data positions: spread over [−1, 1] (the first N of a random set)."
    xdata() = sort(xall[][1:sl_n.value[]])
    noisedata() = eall[][sortperm(xall[][1:sl_n.value[]])]

    "Sampler settings from the sliders (after the optimal values have been written into them)."
    function set_tune!()
        P = prob[]
        if tg_opt.active[]
            s, ε, L = optimal(P, tg_pre.active[])
            syncing[] = true
            set_close_to!(sl_s, s); set_close_to!(sl_eps, ε); set_close_to!(sl_L, L)
            syncing[] = false
        end
        tune[] = Tune(P, tg_pre.active[], sl_s.value[], sl_eps.value[], sl_L.value[])
        T = tune[]
        tunelab.text[] = @sprintf("D = %d,  ω_max/ω_min = %.3g,  Metropolis s = %.3g\nHMC ε = %.3g,  L = %d  (%.2f periods%s)",
                                  P.D, T.ωmax / T.ωmin, T.s, T.ε, T.L, T.L * T.ε * T.ωmin / 2π,
                                  T.ε * T.ωmax >= 2 ? ", UNSTABLE" : "")
    end

    function init!()
        D = sl_D.value[]
        prob[] = Problem(D, kind(), xdata(), noisedata(), sl_sig.value[])
        used[] = (D, kind(), sl_sig.value[], sl_n.value[])
        set_tune!()
        reset!()
        P = prob[]
        o_dat[] = Point2f.(P.x, P.y)
        o_err[] = vec(permutedims(hcat(Point2f.(P.x, P.y .- P.σ), Point2f.(P.x, P.y .+ P.σ))))
        o_true[] = Point2f.(xs, basis(xs, D, P.kind) * CTRUE[1:D])
        lo, hi = extrema(vcat(P.y .- P.σ, P.y .+ P.σ))
        limits!(axD, -1.05, 1.05, lo - 0.1(hi - lo), hi + 0.25(hi - lo))
        axD.title = @sprintf("data: N = %d, σ = %.2g, D = %d (%s)", sl_n.value[], P.σ, D,
                             P.kind === :legendre ? "Legendre" : "monomials")
        draw!()
    end

    function reset!()
        empty!(ch)
        c0 = zeros(prob[].D)
        push!(ch, Chain(c0, prob[]), Chain(c0, prob[]))
    end

    "n HMC iterations and n (or n·(L+1) with 'equal cost') Metropolis iterations."
    function run!(n)
        P = prob[]; T = tune[]
        nm = tg_eq.active[] ? n * (T.L + 1) : n
        c = ch[1]; θ = c.cs[end]; u = c.u
        for _ in 1:nm
            θ, u, acc = mh_step(rng_m, θ, u, T, P)
            c.nacc += acc
            push!(c.cs, θ); push!(c.cost, c.cost[end] + 1)
        end
        c.u = u
        c = ch[2]; θ = c.cs[end]
        for i in 1:n
            ε = tg_jit.active[] ? T.ε * (0.8 + 0.4rand(rng_h)) : T.ε
            θ, acc, div, traj = hmc_step(rng_h, θ, ε, T, P; keep = i == n)
            c.nacc += acc; c.ndiv += div
            i == n && (c.traj = traj)
            push!(c.cs, θ); push!(c.cost, c.cost[end] + T.L + 1)
        end
    end

    "The two coefficients of the projection: j and j+1 (or j−1)."
    function pair()
        D = prob[].D
        j = min(sl_j.value[], D - 1)
        k = j + 1 <= D - 1 ? j + 1 : max(j - 1, 0)
        return j, k
    end

    function draw!()
        P = prob[]; D = P.D
        j, k = pair()
        cm = tg_cost.active[]
        nb = sl_burn.value[]
        pts(c) = Point2f(c[j+1], c[k+1])
        # projection: exact 1σ, 2σ ellipses of the marginal of (c_j, c_k)
        if j != k
            S2 = [P.Σ[j+1, j+1] P.Σ[j+1, k+1]; P.Σ[k+1, j+1] P.Σ[k+1, k+1]]
            Lc = cholesky(Symmetric(S2)).L
            el = Point2f[]
            for r in (1.0, 2.0)
                append!(el, [Point2f((P.m[[j+1, k+1]] .+ r .* (Lc * [cos(t), sin(t)]))...) for t in range(0, 2π, length = 120)])
                push!(el, Point2f(NaN, NaN))
            end
            o_ell[] = el
        else
            o_ell[] = PN
        end
        o_mtrue[] = [Point2f(CTRUE[j+1], CTRUE[k+1])]
        axP.xlabel = "c$(j)"; axP.ylabel = "c$(k)"
        allp = Point2f[]
        xmax = 10.0; emin = Inf; emax = 0.0
        for (i, c) in enumerate(ch)
            n = length(c.cs) - 1
            o_vis[i][]  = pts.(c.cs)
            o_path[i][] = pts.(c.cs[max(1, end - NPATH):end])
            o_cur[i][]  = [pts(c.cs[end])]
            append!(allp, o_path[i][])
            # data panel
            Φx = basis(xs, D, P.kind)
            o_fit[i][] = Point2f.(xs, Φx * c.cs[end])
            sp = Point2f[]
            for s in c.cs[max(1, end - NSPAG + 1):end]
                append!(sp, Point2f.(xs, Φx * s)); push!(sp, Point2f(NaN, NaN))
            end
            o_spag[i][] = sp
            # trace of c_j
            x = cm ? c.cost : collect(0.0:n)
            o_tr[i][] = Point2f.(x, [s[j+1] for s in c.cs])
            xmax = max(xmax, x[end])
            # convergence: Mahalanobis error of the running mean after burn-in, / √D
            if n > nb
                sum_ = zeros(D); N = n - nb
                idx = Set(logidx(N, NPLOT)); out = Point2f[]
                for t in 1:N
                    sum_ .+= c.cs[nb+1+t]
                    if t in idx
                        d = sqrt(maha2(sum_ ./ t, P) / D)
                        push!(out, Point2f(cm ? c.cost[nb+1+t] : nb + t, max(d, 1e-4)))
                        emin = min(emin, d); emax = max(emax, d)
                    end
                end
                o_conv[i][] = out
            else
                o_conv[i][] = Point2f[(NaN, NaN)]
            end
            # table
            q = q997(D)
            r = findfirst(s -> maha2(s, P) < q, c.cs)
            vals = ["$n", @sprintf("%.0f", c.cost[end]),
                    n == 0 ? "–" : @sprintf("%.0f %%", 100c.nacc / n) * (c.ndiv > 0 ? " ($(c.ndiv) div.)" : ""),
                    r === nothing ? "not yet" : @sprintf("it. %d (%.0f ev.)", r - 1, c.cost[r])]
            if n > nb + 1
                S = reduce(hcat, c.cs[nb+2:end])
                sd = sqrt.(diag(P.Σ))
                err = maximum(abs.(vec(sum(S, dims = 2)) ./ size(S, 2) .- P.m) ./ sd)
                e = essmin(S); ev = c.cost[end] - c.cost[nb+1]
                push!(vals, @sprintf("%.2g posterior sd", err), isnan(e) ? "(≥ 50 samples)" : @sprintf("%.0f", e),
                      isnan(e) ? "–" : @sprintf("%.3g", 1000e / ev))
            else
                append!(vals, ["–", "–", "–"])
            end
            for (r_, v) in enumerate(vals)
                cells[r_, i].text[] = v
            end
        end
        tr = ch[2].traj
        o_traj[] = isempty(tr) ? PN : [pts(t) for t in tr if all(isfinite, t)]
        # limits of the projection: posterior ± 4 sd and the recent paths
        sj, sk = sqrt(P.Σ[j+1, j+1]), sqrt(P.Σ[k+1, k+1])
        xl = [P.m[j+1] - 4sj, P.m[j+1] + 4sj]; yl = [P.m[k+1] - 4sk, P.m[k+1] + 4sk]
        for p in allp
            all(isfinite, p) || continue
            xl[1] = min(xl[1], p[1]); xl[2] = max(xl[2], p[1]); yl[1] = min(yl[1], p[2]); yl[2] = max(yl[2], p[2])
        end
        dx = 0.05 * (xl[2] - xl[1]); dy = 0.05 * (yl[2] - yl[1])
        limits!(axP, xl[1] - dx, xl[2] + dx, yl[1] - dy, yl[2] + dy)
        # trace
        allv = vcat([[s[j+1] for s in c.cs] for c in ch]...)
        lo, hi = extrema(vcat(allv, P.m[j+1] - 4sj, P.m[j+1] + 4sj))
        limits!(axT, cm ? -0.01xmax : -0.02xmax, 1.02xmax, lo - 0.05(hi - lo), hi + 0.05(hi - lo))
        axT.xlabel = cm ? "model evaluations" : "iteration"
        axT.ylabel = "c$(j)"; axT.title = "trace of c$(j)  (dashed: exact mean ± sd, dotted: true value)"
        o_band[] = Rect2f(cm ? -0.01xmax : -0.02xmax, P.m[j+1] - sj, 1.05xmax, 2sj)
        o_bm[] = P.m[j+1]; o_bt[] = CTRUE[j+1]
        vis_burn[] = !cm
        o_burn[] = Rect2f(0, lo - hi, min(nb, maximum(length(c.cs) for c in ch) - 1), 3(hi - lo))
        # convergence limits
        axE.xlabel = cm ? "model evaluations" : "iteration"
        vis_ref[] = !cm
        nmax = maximum(length(c.cs) - 1 for c in ch)
        if nmax > nb + 1 && isfinite(emin)
            nr = collect(1:nmax-nb); idx = logidx(length(nr), 400)
            o_ref[] = Point2f.(nb .+ nr[idx], 1 ./ sqrt.(nr[idx]))
            emin = min(emin, 1 / sqrt(nmax - nb)); emax = max(emax, 1.0)
            xhi = max(cm ? xmax : nmax, nb + 2)
            limits!(axE, max(nb + 1, 1), xhi * 1.1, emin / 1.5, emax * 1.5)
        else
            o_ref[] = Point2f[(1, 1)]
            limits!(axE, 1, 10, 0.01, 10)
        end
    end

    "Scan over D with the optimal settings; plot ESS per 1000 evaluations against D."
    function run_scan!()
        B = sl_budget.value[]
        t0 = time()
        res = scan(kind(), tg_pre.active[], xdata(), noisedata(), sl_sig.value[], B, rng_s; jitter = tg_jit.active[])
        scanres[] = res
        for (i, k) in enumerate((:mh, :hmc))
            r = res[k]
            bad  = [Point2f(D, max(r[D, 1], 1e-3)) for D in 1:DMAX if isfinite(r[D, 1]) && r[D, 4] < ESS_FLOOR]
            o_sc[i][] = [Point2f(D, isfinite(r[D, 1]) ? max(r[D, 1], 1e-3) : NaN) for D in 1:DMAX]
            o_scbad[i][] = isempty(bad) ? PN : bad
            o_scbar[i][] = vcat([[Point2f(D, max(r[D, 2], 1e-3)), Point2f(D, max(r[D, 3], 1e-3))] for D in 1:DMAX]...)
            # theoretical trend through the value at D = 1
            if isfinite(r[1, 1])
                Ds = range(1, DMAX, length = 50)
                o_th[i][] = Point2f.(Ds, r[1, 1] .* (k === :mh ? 1 ./ Ds : Ds .^ -0.25))
            end
        end
        v = filter(isfinite, vcat(res[:mh][:, 2:3][:], res[:hmc][:, 2:3][:]))
        lo = max(minimum(v; init = 1.0), 1e-3); hi = maximum(v; init = 100.0)
        ylims!(axS, lo / 2, hi * 3)
        axS.title = @sprintf("scan: %s, %s, %d ev. × %d runs", kind() === :legendre ? "Legendre" : "monomials",
                             tg_pre.active[] ? "precond." : "C = I", B, NREP)
        m, h = res[:mh], res[:hmc]
        set_status!(@sprintf("Scan done in %.1f s (true ESS from %d runs per D). ESS per 1000 evaluations, D = 1: Metropolis %.3g, HMC %.3g;  D = %d: Metropolis %.3g, HMC %.3g  (HMC/Metropolis = %.2g). Hollow: ESS < %d per run (chain hardly moved).",
                             time() - t0, NREP, m[1, 1], h[1, 1], DMAX, m[DMAX, 1], h[DMAX, 1], h[DMAX, 1] / m[DMAX, 1], ESS_FLOOR))
    end

    data_pending() = (sl_D.value[], kind(), sl_sig.value[], sl_n.value[]) != used[]

    # ------------------------------------------------------------ callbacks
    on(_ -> (init!(); set_status!("Data initialised. Both chains start at c = 0.")), btn_init.clicks)
    on(btn_noise.clicks) do _
        new_noise!(); init!(); set_status!("New noise realisation (new x positions and noise).")
    end
    on(sl_D.value) do _
        init!()
        set_status!("D = $(sl_D.value[]) coefficients: new problem, chains restarted" *
                    (tg_opt.active[] ? ", optimal settings applied." : "."))
    end
    on(_ -> (init!(); set_status!("Basis changed: new problem, chains restarted.")), menu.selection)
    for sl in (sl_sig, sl_n)
        on(_ -> data_pending() && set_status!("Data settings changed – press \"Initialise data\" to apply them.", warn = true), sl.value)
    end
    on(_ -> draw!(), sl_j.value)
    on(_ -> draw!(), sl_burn.value)
    on(_ -> draw!(), tg_cost.active)
    on(tg_pre.active) do p
        set_tune!(); reset!(); draw!()
        set_status!(p ? "Preconditioning ON: C = Σ (Metropolis proposal s²Σ, HMC mass matrix Σ⁻¹). Chains restarted." :
                        "Preconditioning OFF: C = I. Chains restarted.")
    end
    on(tg_opt.active) do o
        set_tune!(); o && (reset!(); draw!())
    end
    for sl in (sl_s, sl_eps, sl_L)
        on(sl.value) do _
            syncing[] && return
            if tg_opt.active[]
                syncing[] = true; tg_opt.active[] = false; syncing[] = false
            end
            set_tune!()
        end
    end
    on(_ -> (reset!(); draw!(); set_status!("Both chains restarted at c = 0.")), btn_reset.clicks)
    on(btn_run.clicks) do _
        run!(sl_batch.value[]); draw!()
        data_pending() && set_status!("Note: the data sliders were changed but not applied (press \"Initialise data\").",
                                      warn = true)
    end
    on(_ -> run_scan!(), btn_scan.clicks)

    # ------------------------------------------------------- initialisation
    rng_d[] = Xoshiro(SEED)
    xall[]  = 2 .* rand(rng_d[], NMAX) .- 1
    eall[]  = randn(rng_d[], NMAX)
    init!()
    set_status!("Choose D and press \"Run both\" (Metropolis gets L + 1 iterations per HMC iteration: equal cost), " *
                "or \"Scan D = 1 … 10\" to measure the efficiency of both samplers for every D.")
    return fig, (init = init!, run = n -> (run!(n); draw!()), reset = () -> (reset!(); draw!()), scan = run_scan!,
                 sliders = (D = sl_D, sig = sl_sig, n = sl_n, j = sl_j, s = sl_s, eps = sl_eps, L = sl_L,
                            batch = sl_batch, burn = sl_burn, budget = sl_budget),
                 toggles = (cost = tg_cost, path = tg_path, spag = tg_spag, pre = tg_pre, opt = tg_opt, jit = tg_jit, eq = tg_eq),
                 menu = menu, chains = ch, cells = cells, prob = prob, tune = tune, scanres = scanres)
end

# =============================================================================
# RUN
# =============================================================================
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    fig, app = build_app()
    screen = display(fig)
    isinteractive() || wait(screen)     # keep the window open when run from a terminal
end
