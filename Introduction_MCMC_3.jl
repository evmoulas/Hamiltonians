#=
Introduction_MCMC_3 – Metropolis–Hastings for a 2-parameter fit  y = a + k·x
============================================================================

Data  yᵢ = a_true + k_true·xᵢ + σ·εᵢ ,  model  y = a + k·x  (intercept a, slope k).

  misfit      χ²(k, a) = Σᵢ (yᵢ − a − k·xᵢ)² / σ²
  priors      k ~ N(μ_k, σ_k²),  a ~ N(μ_a, σ_a²)   (independent)
  posterior   p(k, a | d) ∝ exp(−χ²/2) · p(k) · p(a)

New in two dimensions: the parameters are CORRELATED. With x in [0, 10] a
larger slope can be compensated by a smaller intercept, so the posterior is a
narrow, tilted ellipse (correlation ≈ −0.8 for the default data). A proposal that ignores this
shape moves slowly along the ellipse.

Metropolis–Hastings (θ = (k, a)):  θ′ ~ q(θ′|θ),
     α = min(1,  p(θ′|d)/p(θ|d) · q(θ|θ′)/q(θ′|θ) ).
Proposals (s = step size):
  1. isotropic random walk         θ′ ~ N(θ, s²·I)                 symmetric
  2. one parameter at a time       k and a are updated alternately symmetric
  3. random walk with covariance   θ′ ~ N(θ, s²·Σ̂)                symmetric
  4. independence sampler          θ′ ~ N(θ̄, s²·Σ̂)                needs the Hastings ratio
  5. Langevin (MALA)               θ′ ~ N(θ + ½s²∇ln p, s²·I)      needs the Hastings ratio
  θ̄ and Σ̂ are the mean and covariance of the samples (after burn-in) at the
  moment "learn Σ̂ from chain" is pressed (adaptive MCMC in two stages: learn
  during a pilot run, then keep Σ̂ fixed). For proposal 3 the classical
  optimal step is s ≈ 2.4/√2 ≈ 1.7, i.e. 1.5 – 2 on the slider (acceptance ≈ 35 %).

Panels
  upper left   : data, true line, current fit and the last states of the chain
  upper right  : corner plot – the misfit map χ²(k, a) (log shading, Δχ² contours
                 2.3 / 6.2 / 11.8 = 68 / 95 / 99.7 % for two parameters) with the
                 chain, the proposal density q(·|θ) and the exact posterior
                 ellipses; on top the marginal of k, on the right the marginal of a
  lower right  : trace plots of k and a
  lower left   : controls; click in the χ² map to set the starting point

What to try
  * proposal 1 with s = 0.1: the chain crawls along the valley; ESS is small.
  * proposal 2: also slow – axis-parallel steps in a tilted valley.
  * run a pilot (e.g. 2000 samples), press "learn Σ̂", switch to proposal 3 with
    s ≈ 1.5: about 7× larger ESS for the same number of samples (proposal 4: ≈ 30×).
  * proposals 4 and 5 with the Hastings correction OFF: wrong posterior.
  * proposal 4 with a narrow q (s < 1) started far from the posterior: the chain
    never moves. There q(θ) is tiny compared with p(θ|d), so the Hastings ratio
    q(θ)/q(θ′) ≈ 0 rejects every proposal. An independence proposal must be
    WIDER than the posterior (s ≥ 1.5) – its tails must not be too light.

Run from a terminal:      julia Introduction_MCMC_3.jl
or from the Julia REPL:   include("Introduction_MCMC_3.jl")
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
const KLIM     = (-1.0, 4.0)        # k range of the misfit map (full view)
const ALIM     = (-8.0, 8.0)        # a range of the misfit map (full view)
const NG       = 201                # grid (NG × NG) of the misfit map
const NMAX     = 50                 # largest number of data points
const SEED     = 1                  # seed of the first noise realisation

const K_TRUE   = 1.5                # default true slope
const A_TRUE   = 2.0                # default true intercept
const SIGMA    = 2.0                # default noise standard deviation
const NDATA    = 15                 # default number of data points
const START    = (-0.5, -5.0)       # default starting point (k₀, a₀)
const PRIOR_MK = 0.5;  const PRIOR_SK = 2.0     # default prior of k
const PRIOR_MA = 0.0;  const PRIOR_SA = 5.0     # default prior of a
const STEP     = 0.1                # default step size s
const BATCHES  = [1, 10, 100, 1000] # batch sizes of the "Run" button
const NSPAG    = 40                 # number of last chain states drawn in the data panel
const NPATH    = 150                # number of last steps drawn as a path in the misfit map
const DCHI     = [2.30, 6.18, 11.8] # Δχ² of the 68 / 95 / 99.7 % regions (2 parameters)

const PROPOSALS = ["1 · isotropic random walk   N(θ, s²I)",
                   "2 · one parameter at a time   (k, a, k, …)",
                   "3 · random walk with covariance   N(θ, s²Σ)",
                   "4 · independence sampler   N(mean, s²Σ)",
                   "5 · Langevin (MALA)   N(θ + ½s²∇ln p, s²I)"]

const C_DATA  = :black
const C_TRUE  = :gray40
const C_CUR   = :darkorange2        # current state
const C_CHAIN = :seagreen           # visited states / histograms
const C_REJ   = :red3               # rejected proposals
const C_PRIOR = :royalblue3
const C_POST  = :black              # exact posterior
const C_MIS   = :purple4            # misfit contours
const C_Q     = :darkorange3        # proposal density

# =============================================================================
# MATHEMATICS (independent of the plotting)
# =============================================================================
"Symmetric 2×2 matrix [xx xy; xy yy] (covariances, precisions); index 1 = k, 2 = a."
struct Sym2
    xx::Float64
    xy::Float64
    yy::Float64
end
const I2 = Sym2(1.0, 0.0, 1.0)
Base.:*(c::Real, S::Sym2) = Sym2(c * S.xx, c * S.xy, c * S.yy)
Base.:+(A::Sym2, B::Sym2) = Sym2(A.xx + B.xx, A.xy + B.xy, A.yy + B.yy)
det2(S::Sym2) = S.xx * S.yy - S.xy^2
inv2(S::Sym2) = (1 / det2(S)) * Sym2(S.yy, -S.xy, S.xx)
mul(S::Sym2, v) = (S.xx * v[1] + S.xy * v[2], S.xy * v[1] + S.yy * v[2])
quad(S::Sym2, v) = v[1] * (S.xx * v[1] + S.xy * v[2]) + v[2] * (S.xy * v[1] + S.yy * v[2])
"Lower Cholesky factor (l11, l21, l22) of S."
function chol2(S::Sym2)
    l11 = sqrt(S.xx); l21 = S.xy / l11
    return l11, l21, sqrt(max(S.yy - l21^2, 0.0))
end
sd2(S::Sym2) = (sqrt(S.xx), sqrt(S.yy))
corr2(S::Sym2) = S.xy / sqrt(S.xx * S.yy)

"Sums of the data: N, Σx, Σx², Σy, Σxy, Σy² and σ²."
struct DataSums
    n::Float64; x::Float64; xx::Float64; y::Float64; xy::Float64; yy::Float64; s2::Float64
end
DataSums(x, y, σ) = DataSums(length(x), sum(x), sum(abs2, x), sum(y), sum(x .* y), sum(abs2, y), σ^2)

"Misfit χ²(k, a) = Σ (yᵢ − a − k xᵢ)² / σ²  (expanded: O(1))."
chi2(k, a, S::DataSums) =
    max(S.yy - 2a * S.y - 2k * S.xy + S.n * a^2 + 2a * k * S.x + k^2 * S.xx, 0.0) / S.s2

"Priors: means (μ_k, μ_a) and standard deviations (σ_k, σ_a)."
struct Prior
    mk::Float64; sk::Float64; ma::Float64; sa::Float64
end
logprior(k, a, P::Prior) = -0.5 * ((k - P.mk) / P.sk)^2 - 0.5 * ((a - P.ma) / P.sa)^2
logpost(k, a, S, P) = -0.5 * chi2(k, a, S) + logprior(k, a, P)
"Gradient (∂/∂k, ∂/∂a) of ln p(k, a | d)."
gradlogpost(k, a, S, P) = ((S.xy - a * S.x - k * S.xx) / S.s2 - (k - P.mk) / P.sk^2,
                           (S.y - S.n * a - k * S.x) / S.s2 - (a - P.ma) / P.sa^2)

"Exact Gaussian posterior: mean (k, a) and covariance."
function exact_posterior(S::DataSums, P::Prior)
    Λ = Sym2(S.xx / S.s2 + 1 / P.sk^2, S.x / S.s2, S.n / S.s2 + 1 / P.sa^2)
    C = inv2(Λ)
    return mul(C, (S.xy / S.s2 + P.mk / P.sk^2, S.y / S.s2 + P.ma / P.sa^2)), C
end
"Least squares (minimum of χ²): (k̂, â) and their covariance."
function lsq(S::DataSums)
    C = inv2(Sym2(S.xx / S.s2, S.x / S.s2, S.n / S.s2))
    return mul(C, (S.xy / S.s2, S.y / S.s2)), C
end

gauss(x, m, s) = exp(-0.5 * ((x - m) / s)^2) / (s * sqrt(2π))

"Points on the ellipse  (θ − m)ᵀ C⁻¹ (θ − m) = r²  (a NaN at the end separates curves)."
function ellipse(m, C::Sym2, r; n = 120)
    l11, l21, l22 = chol2(C)
    pts = [Point2f(m[1] + r * l11 * cos(t), m[2] + r * (l21 * cos(t) + l22 * sin(t))) for t in range(0, 2π, length = n)]
    push!(pts, Point2f(NaN, NaN))
    return pts
end

"Settings of the sampler."
struct Sampler
    type::Int
    s::Float64
    hastings::Bool
    S::DataSums
    P::Prior
    mbar::NTuple{2,Float64}      # θ̄ (independence sampler)
    C::Sym2                      # Σ̂
end

"Centre and covariance of the Gaussian proposal q(·|θ) (types 1, 3, 4, 5)."
function qpars(Q::Sampler, θ)
    Q.type == 1 && return θ, Q.s^2 * I2
    Q.type == 3 && return θ, Q.s^2 * Q.C
    Q.type == 4 && return Q.mbar, Q.s^2 * Q.C
    g = gradlogpost(θ..., Q.S, Q.P)                         # Langevin
    return (θ[1] + 0.5Q.s^2 * g[1], θ[2] + 0.5Q.s^2 * g[2]), Q.s^2 * I2
end
"ln q(θ′|θ) + const."
function logq(Q::Sampler, θp, θ)
    c, C = qpars(Q, θ)
    return -0.5 * quad(inv2(C), (θp[1] - c[1], θp[2] - c[2]))
end

"""
One Metropolis–Hastings step from θ (ln posterior lp); `it` = iteration number
(proposal 2 updates k for odd and a for even iterations).
"""
function mh_step(rng, θ, lp, Q::Sampler, it)
    if Q.type == 2
        j  = isodd(it) ? 1 : 2
        θp = j == 1 ? (θ[1] + Q.s * randn(rng), θ[2]) : (θ[1], θ[2] + Q.s * randn(rng))
        lH = 0.0
    else
        c, C = qpars(Q, θ)
        l11, l21, l22 = chol2(C)
        z1, z2 = randn(rng), randn(rng)
        θp = (c[1] + l11 * z1, c[2] + l21 * z1 + l22 * z2)
        lH = (Q.type in (1, 3)) ? 0.0 : logq(Q, θ, θp) - logq(Q, θp, θ)
    end
    lpp = logpost(θp..., Q.S, Q.P)
    lA  = lpp - lp
    lα  = min(0.0, lA + (Q.hastings ? lH : 0.0))
    u   = rand(rng)
    return (θ_old = θ, θp = θp, lpp = lpp, A = exp(lA), H = exp(lH), α = exp(lα), u = u, acc = log(u) < lα)
end

"Density histogram with about √n bins (at most 50). Returns (centres, densities, width)."
function density_hist(v)
    n = length(v)
    n == 0 && return [0.0], [0.0], 1.0
    lo, hi = extrema(v)
    if hi - lo < 1e-12
        w = 0.01 * max(abs(lo), 1.0)
        return [lo], [1 / w], w
    end
    nb = clamp(round(Int, sqrt(n)), 5, 50)
    w  = (hi - lo) / nb
    c  = zeros(nb)
    @inbounds for x in v
        c[clamp(floor(Int, (x - lo) / w) + 1, 1, nb)] += 1
    end
    return collect(lo .+ w .* ((1:nb) .- 0.5)), c ./ (n * w), w
end

mean_(v) = sum(v) / length(v)
var_(v) = (m = mean_(v); sum(abs2, v .- m) / (length(v) - 1))
"Effective sample size by batch means (≈ n / integrated autocorrelation time)."
function ess(v)
    n = length(v)
    n < 50 && return NaN
    b = floor(Int, sqrt(n)); nb = n ÷ b
    bm = [mean_(view(v, (i-1)*b+1:i*b)) for i in 1:nb]
    vb = var_(bm); vx = var_(v)
    (vb <= 0 || vx <= 0) && return NaN
    return min(n, n * vx / (b * vb))
end

fmt(v) = @sprintf("%.4g", abs(v) < 1e-9 ? 0.0 : v)
fmt2(v) = @sprintf("%.3g", abs(v) < 1e-9 ? 0.0 : v)
fmtr(v) = (v == 0 || 1e-3 <= abs(v) < 1e4) ? @sprintf("%.3g", v) : @sprintf("%.2e", v)

# =============================================================================
# APPLICATION
# =============================================================================
function build_app()
    # ---------------------------------------------------------------- state
    rng_d  = Ref(Xoshiro(SEED))                       # data (x positions and noise)
    rng_c  = Xoshiro(SEED + 1000)                     # Markov chain
    xall   = Ref(zeros(NMAX)); eall = Ref(zeros(NMAX))
    xd     = Ref(Float64[]);   yd   = Ref(Float64[])
    S      = Ref(DataSums(2, 1, 1, 0, 0, 0, 1))
    used   = Ref((K_TRUE, A_TRUE, SIGMA, NDATA))      # data settings in use
    start  = Ref(START)
    ks     = Float64[]; as = Float64[]                # visited states (index 1 = start)
    lps    = Float64[]
    props  = NTuple{2,Float64}[]; accs = Bool[]
    laststep = Ref{Any}(nothing); lastn = Ref(0)
    ptype  = Ref(1)
    learned = Ref((START, I2, false))                 # (θ̄, Σ̂, learned?)

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Introduction to MCMC 3  –  Metropolis–Hastings for two parameters  (Bayesian fit of  y = a + k·x)",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(540))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")        # no fixed height: it takes the space the controls leave

    corner = right[1, 1] = GridLayout()
    axK = Axis(corner[1, 1]; ylabel = "p(k|d)", xticklabelsvisible = false)
    axM = Axis(corner[2, 1]; xlabel = "slope k", ylabel = "intercept a")
    axA = Axis(corner[2, 2]; xlabel = "p(a|d)", yticklabelsvisible = false, xticks = WilkinsonTicks(3; k_min = 2))
    rowsize!(corner, 1, Relative(0.27)); colsize!(corner, 2, Relative(0.2))
    rowgap!(corner, 6); colgap!(corner, 6)
    axTk = Axis(right[2, 1]; ylabel = "k", xticklabelsvisible = false)
    axTa = Axis(right[3, 1]; ylabel = "a", xlabel = "iteration")
    rowsize!(right, 1, Auto(3.2)); rowgap!(right, 6)
    for a in (axD, axK, axM, axA, axTk, axTa), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
    end

    # display options (below the trace plots)
    tg = right[4, 1] = GridLayout(halign = :center, tellwidth = false)
    tg_zoom  = Toggle(tg[1, 1], active = false); Label(tg[1, 2], "zoom on posterior", halign = :left)
    tg_exact = Toggle(tg[1, 3], active = true);  Label(tg[1, 4], "exact posterior", halign = :left)
    tg_prior = Toggle(tg[1, 5], active = true);  Label(tg[1, 6], "prior", halign = :left)
    tg_rej   = Toggle(tg[2, 1], active = true);  Label(tg[2, 2], "rejected proposals", halign = :left)
    tg_q     = Toggle(tg[2, 3], active = true);  Label(tg[2, 4], "proposal q(·|θ)", halign = :left)
    tg_path  = Toggle(tg[2, 5], active = true);  Label(tg[2, 6], "chain path", halign = :left)
    colgap!(tg, 6); rowgap!(tg, 2)
    for c in (2, 4)
        colgap!(tg, c, 18)
    end

    # --- 1. data
    Label(left[2, 1], "1 · synthetic data   y = a_true + k_true·x + σ·ε", font = :bold, halign = :left, tellwidth = false)
    nice = sort(unique(round.(vcat([m .* 10.0^e for e in -3:2 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3)))
    sgD = SliderGrid(left[3, 1],
        (label = "k_true", range = 0:0.05:3, startvalue = K_TRUE, format = "{:.2f}"),
        (label = "a_true", range = -5:0.1:5, startvalue = A_TRUE, format = "{:.1f}"),
        (label = "noise σ", range = 0.1:0.1:5, startvalue = SIGMA, format = v -> @sprintf("%.1f  (σ² = %.3g)", v, v^2)),
        (label = "N points", range = 3:NMAX, startvalue = NDATA, format = "{:d}");
        tellheight = true, width = 500, halign = :left)
    sl_kt, sl_at, sl_sig, sl_n = sgD.sliders
    bD = left[4, 1] = GridLayout(halign = :left)
    btn_init  = Button(bD[1, 1], label = "Initialise data (compute χ² map)", buttoncolor = RGBf(0.80, 0.88, 1.0))
    btn_noise = Button(bD[1, 2], label = "new noise realisation")

    # --- 2. priors and start
    Label(left[5, 1], "2 · priors   k ~ N(μ_k, σ_k²),   a ~ N(μ_a, σ_a²)", font = :bold, halign = :left, tellwidth = false)
    sgP = SliderGrid(left[6, 1],
        (label = "prior mean μ_k", range = KLIM[1]:0.05:KLIM[2], startvalue = PRIOR_MK, format = "{:.2f}"),
        (label = "prior st.dev. σ_k", range = filter(v -> 0.02 <= v <= 20, nice), startvalue = PRIOR_SK, format = "{:.3g}"),
        (label = "prior mean μ_a", range = ALIM[1]:0.1:ALIM[2], startvalue = PRIOR_MA, format = "{:.1f}"),
        (label = "prior st.dev. σ_a", range = filter(v -> 0.1 <= v <= 50, nice), startvalue = PRIOR_SA, format = "{:.3g}");
        tellheight = true, width = 500, halign = :left)
    sl_mk, sl_sk, sl_ma, sl_sa = sgP.sliders
    startlabel = Label(left[7, 1], " ", halign = :left, tellwidth = false, color = :gray30, fontsize = 13)

    # --- 3. sampler
    Label(left[8, 1], "3 · Metropolis–Hastings   α = min(1, p(θ′|d)/p(θ|d) · q(θ|θ′)/q(θ′|θ))", font = :bold,
          halign = :left, tellwidth = false)
    mg = left[9, 1] = GridLayout(halign = :left)
    menu = Menu(mg[1, 1], options = PROPOSALS, default = PROPOSALS[1], width = 330)
    tg_hast = Toggle(mg[1, 2], active = true)
    Label(mg[1, 3], "Hastings\ncorrection", halign = :left, justification = :left, fontsize = 12)
    colgap!(mg, 1, 12); colgap!(mg, 2, 4)
    sgS = SliderGrid(left[10, 1],
        (label = "step size s", range = filter(v -> 0.001 <= v <= 10, nice), startvalue = STEP, format = "{:.3g}"),
        (label = "batch size", range = BATCHES, startvalue = BATCHES[1], format = "{:d} samples"),
        (label = "burn-in", range = 0:10:5000, startvalue = 0, format = "discard first {:d}");
        tellheight = true, width = 500, halign = :left)
    sl_step, sl_batch, sl_burn = sgS.sliders
    bS = left[11, 1] = GridLayout(halign = :left)
    btn_run   = Button(bS[1, 1], label = "Run MCMC batch", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 170)
    btn_reset = Button(bS[1, 2], label = "Reset chain")
    btn_learn = Button(bS[1, 3], label = "learn mean, Σ from chain", buttoncolor = RGBf(1.0, 0.90, 0.78))

    # fixed heights (6 and 2 lines), so that the data axis does not jump when the text changes
    info   = Label(left[12, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   fontsize = 12, height = 92)
    status = Label(left[13, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 34)
    rowgap!(left, 4)
    for sg in (sgD, sgP, sgS)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    P0 = Point2f[(0, 0)]; PN = Point2f[(NaN, NaN)]
    # data panel
    o_dat  = Observable(P0); o_err = Observable(Point2f[(0, 0), (0, 0)])
    o_true = Observable(Point2f[(0, 0), (1, 0)]); o_spag = Observable(Point2f[(0, 0), (0, 0)])
    o_cur  = Observable(Point2f[(0, 0), (1, 0)])
    o_prop = Observable(Point2f[(0, 0), (1, 0)]); c_prop = Observable{Any}(C_CHAIN); vis_prop = Observable(false)
    # misfit map
    o_gx = Observable(collect(range(0, 1, length = NG))); o_gy = Observable(collect(range(0, 1, length = NG)))
    o_gz = Observable(zeros(NG, NG)); o_gl = Observable(zeros(NG, NG))
    o_lev1 = Observable(collect(1.0:3)); o_lev2 = Observable(collect(1.0:10))
    o_true2 = Observable(P0); o_lsq = Observable(P0)
    o_pell = Observable(PN); o_xell = Observable(PN); o_qell = Observable(PN)
    o_vis  = Observable(P0); o_path = Observable(PN); o_rej = Observable(PN)
    o_mcur = Observable(P0); o_mprop = Observable(PN); o_arrow = Observable(Point2f[(NaN, NaN), (NaN, NaN)])
    # marginals
    o_hk = Observable(P0); o_bwk = Observable(1.0); o_ha = Observable(P0); o_bwa = Observable(1.0)
    vis_hist = Observable(false)
    o_pk = Observable(P0); o_ek = Observable(P0); o_pa = Observable(P0); o_ea = Observable(P0)
    o_kt = Observable(K_TRUE); o_at = Observable(A_TRUE)
    # traces
    o_trk = Observable(P0); o_tra = Observable(P0)
    o_burnk = Observable(Rect2f(0, 0, 0, 1)); o_burna = Observable(Rect2f(0, 0, 0, 1))
    o_bandk = Observable(Rect2f(0, 0, 1, 1)); o_banda = Observable(Rect2f(0, 0, 1, 1))
    o_mk = Observable(0.0); o_ma = Observable(0.0)

    # upper left: data
    lines!(axD, o_true; color = C_TRUE, linewidth = 2, linestyle = :dash, label = "true line")
    linesegments!(axD, o_spag; color = (C_CHAIN, 0.25), linewidth = 1, label = "last $NSPAG states")
    lines!(axD, o_prop; color = c_prop, linewidth = 2, linestyle = :dot, visible = vis_prop, label = "proposal θ′")
    lines!(axD, o_cur; color = C_CUR, linewidth = 3, label = "current fit  y = a + k·x")
    linesegments!(axD, o_err; color = (C_DATA, 0.6), linewidth = 1.2)
    scatter!(axD, o_dat; color = C_DATA, markersize = 9, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 11, rowgap = 0, colgap = 6, nbanks = 2, patchsize = (16, 10),
               padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

    # misfit map
    heatmap!(axM, o_gx, o_gy, o_gl; colormap = :Blues, alpha = 0.75)
    contour!(axM, o_gx, o_gy, o_gz; levels = o_lev2, color = (:gray40, 0.45), linewidth = 0.7)
    contour!(axM, o_gx, o_gy, o_gz; levels = o_lev1, color = C_MIS, linewidth = 1.8)
    lines!(axM, o_pell; color = (C_PRIOR, 0.8), linewidth = 1.5, linestyle = :dash, visible = tg_prior.active)
    lines!(axM, o_xell; color = C_POST, linewidth = 1.8, linestyle = :dash, visible = tg_exact.active)
    scatter!(axM, o_rej; color = (C_REJ, 0.35), marker = :xcross, markersize = 7, visible = tg_rej.active)
    scatter!(axM, o_vis; color = (C_CHAIN, 0.35), markersize = 5)
    lines!(axM, o_path; color = (C_CHAIN, 0.9), linewidth = 1.3, visible = tg_path.active)
    lines!(axM, o_qell; color = C_Q, linewidth = 2, visible = tg_q.active)
    lines!(axM, o_arrow; color = c_prop, linewidth = 2, visible = vis_prop)
    scatter!(axM, o_mprop; color = c_prop, marker = :diamond, markersize = 14, strokecolor = :black, strokewidth = 1,
             visible = vis_prop)
    scatter!(axM, o_true2; color = :white, marker = :star5, markersize = 18, strokecolor = :black, strokewidth = 1.2)
    scatter!(axM, o_lsq; color = C_MIS, marker = :cross, markersize = 14)
    scatter!(axM, o_mcur; color = C_CUR, markersize = 15, strokecolor = :black, strokewidth = 1.5)

    # legend in the empty corner (top right)
    Legend(corner[1, 2],
           [LineElement(color = C_MIS, linewidth = 1.8), LineElement(color = C_POST, linestyle = :dash),
            LineElement(color = C_PRIOR, linestyle = :dash), LineElement(color = C_Q, linewidth = 2),
            MarkerElement(color = (C_CHAIN, 0.7), marker = :circle, markersize = 7),
            MarkerElement(color = (C_REJ, 0.6), marker = :xcross, markersize = 8),
            MarkerElement(color = C_CUR, marker = :circle, markersize = 11, strokecolor = :black, strokewidth = 1),
            MarkerElement(color = :white, marker = :star5, markersize = 12, strokecolor = :black, strokewidth = 1),
            MarkerElement(color = C_MIS, marker = :cross, markersize = 11)],
           ["Δχ² = 2.3, 6.2, 11.8", "exact posterior 1σ, 2σ", "prior 1σ, 2σ", "proposal q 1σ, 2σ",
            "visited states", "rejected", "current state", "true (k, a)", "least squares"];
           labelsize = 10, rowgap = -2, padding = (4, 4, 2, 2), framevisible = false, patchsize = (14, 8),
           tellheight = false, tellwidth = false, valign = :center, halign = :left)

    # marginals
    barplot!(axK, o_hk; width = o_bwk, gap = 0, color = (C_CHAIN, 0.45), strokecolor = (C_CHAIN, 0.9),
             strokewidth = 0.5, visible = vis_hist)
    lines!(axK, o_pk; color = C_PRIOR, linewidth = 2, visible = tg_prior.active)
    lines!(axK, o_ek; color = C_POST, linewidth = 1.8, linestyle = :dash, visible = tg_exact.active)
    vlines!(axK, o_kt; color = C_TRUE, linestyle = :dash, linewidth = 1.2)
    barplot!(axA, o_ha; width = o_bwa, gap = 0, direction = :x, color = (C_CHAIN, 0.45), strokecolor = (C_CHAIN, 0.9),
             strokewidth = 0.5, visible = vis_hist)
    lines!(axA, o_pa; color = C_PRIOR, linewidth = 2, visible = tg_prior.active)
    lines!(axA, o_ea; color = C_POST, linewidth = 1.8, linestyle = :dash, visible = tg_exact.active)
    hlines!(axA, o_at; color = C_TRUE, linestyle = :dash, linewidth = 1.2)

    # traces
    for (ax, band, m, burn, tr, tv) in ((axTk, o_bandk, o_mk, o_burnk, o_trk, o_kt), (axTa, o_banda, o_ma, o_burna, o_tra, o_at))
        poly!(ax, band; color = (C_POST, 0.08), visible = tg_exact.active)
        hlines!(ax, m; color = C_POST, linestyle = :dash, linewidth = 1.2, visible = tg_exact.active)
        poly!(ax, burn; color = (:gray50, 0.18))
        hlines!(ax, tv; color = C_TRUE, linestyle = :dash, linewidth = 1.5)
        lines!(ax, tr; color = C_CHAIN, linewidth = 1)
    end

    # ------------------------------------------------------------ functions
    prior() = Prior(sl_mk.value[], sl_sk.value[], sl_ma.value[], sl_sa.value[])
    sampler() = Sampler(ptype[], sl_step.value[], tg_hast.active[], S[], prior(), learned[][1], learned[][2])
    nit() = length(ks) - 1

    function new_noise!()
        rng_d[] = Xoshiro(rand(UInt32))
        xall[]  = XLIM[1] .+ (XLIM[2] - XLIM[1]) .* rand(rng_d[], NMAX)
        eall[]  = randn(rng_d[], NMAX)
    end

    "Data from the sliders and the stored noise."
    function init_data!()
        kt, at, σ, n = sl_kt.value[], sl_at.value[], sl_sig.value[], sl_n.value[]
        used[] = (kt, at, σ, n)
        xd[] = xall[][1:n]
        yd[] = at .+ kt .* xd[] .+ σ .* eall[][1:n]
        S[]  = DataSums(xd[], yd[], σ)
        o_kt[] = kt; o_at[] = at
        o_true2[] = [Point2f(kt, at)]
        o_lsq[]   = [Point2f(lsq(S[])[1]...)]
        o_dat[]  = Point2f.(xd[], yd[])
        o_err[]  = vec(permutedims(hcat(Point2f.(xd[], yd[] .- σ), Point2f.(xd[], yd[] .+ σ))))
        o_true[] = [Point2f(XLIM[1], at + kt * XLIM[1]), Point2f(XLIM[2], at + kt * XLIM[2])]
        axD.title = @sprintf("data: N = %d,  k_true = %.2f,  a_true = %.1f,  σ = %.2g", n, kt, at, σ)
        lo, hi = extrema(vcat(yd[] .- σ, yd[] .+ σ, 0.0))
        limits!(axD, XLIM[1] - 0.3, XLIM[2] + 0.3, lo - 0.08(hi - lo), hi + 0.08(hi - lo))
    end

    "Window of the parameter axes: (k1, k2, a1, a2)."
    function window()
        tg_zoom.active[] || return (KLIM..., ALIM...)
        m, C = exact_posterior(S[], prior()); sk, sa = sd2(C)
        return (m[1] - 4.5sk, m[1] + 4.5sk, m[2] - 4.5sa, m[2] + 4.5sa)
    end

    "The misfit map χ²(k, a) on the grid of the current window (log shading + Δχ² contours)."
    function draw_map!()
        k1, k2, a1, a2 = window()
        gx = collect(range(k1, k2, length = NG)); gy = collect(range(a1, a2, length = NG))
        Sd = S[]
        gz = [chi2(k, a, Sd) for k in gx, a in gy]
        cmin = chi2(lsq(Sd)[1]..., Sd)
        o_gx[] = gx; o_gy[] = gy                          # sizes never change: no size mismatch
        o_lev1[] = cmin .+ DCHI
        o_lev2[] = cmin .+ 25 .* 2.0 .^ (0:14)
        o_gl[] = log10.(gz .- cmin .+ 1)
        o_gz[] = gz
        limits!(axM, k1, k2, a1, a2)
    end

    "Prior ellipses, exact posterior ellipses and marginal curves."
    function draw_prior!()
        P = prior(); m, C = exact_posterior(S[], P)
        o_pell[] = vcat(ellipse((P.mk, P.ma), Sym2(P.sk^2, 0.0, P.sa^2), 1.0), ellipse((P.mk, P.ma), Sym2(P.sk^2, 0.0, P.sa^2), 2.0))
        o_xell[] = vcat(ellipse(m, C, 1.0), ellipse(m, C, 2.0))
        k1, k2, a1, a2 = window(); sk, sa = sd2(C)
        kk = sort(vcat(range(min(k1, KLIM[1]), max(k2, KLIM[2]), length = 2001), range(m[1] - 6sk, m[1] + 6sk, length = 601)))
        aa = sort(vcat(range(min(a1, ALIM[1]), max(a2, ALIM[2]), length = 2001), range(m[2] - 6sa, m[2] + 6sa, length = 601)))
        o_pk[] = Point2f.(kk, gauss.(kk, P.mk, P.sk)); o_ek[] = Point2f.(kk, gauss.(kk, m[1], sk))
        o_pa[] = Point2f.(gauss.(aa, P.ma, P.sa), aa); o_ea[] = Point2f.(gauss.(aa, m[2], sa), aa)
        o_mk[] = m[1]; o_ma[] = m[2]
    end

    function reset_chain!()
        empty!(ks); empty!(as); empty!(lps); empty!(props); empty!(accs)
        θ0 = start[]
        push!(ks, θ0[1]); push!(as, θ0[2]); push!(lps, logpost(θ0..., S[], prior()))
        laststep[] = nothing; lastn[] = 0
        startlabel.text[] = @sprintf("start (k₀, a₀) = (%.2f, %.2f)  –  click in the χ² map to change it", θ0...)
    end

    function run_batch!(n)
        Q = sampler()
        θ, lp = (ks[end], as[end]), lps[end]
        for _ in 1:n
            r = mh_step(rng_c, θ, lp, Q, nit() + 1)
            push!(props, r.θp); push!(accs, r.acc)
            r.acc && ((θ, lp) = (r.θp, r.lpp))
            push!(ks, θ[1]); push!(as, θ[2]); push!(lps, lp)
            laststep[] = r
        end
        lastn[] = n
    end

    "Samples after the burn-in (index 1 is the start, iteration 0)."
    function kept()
        nb = sl_burn.value[]
        nit() > nb ? (ks[nb+2:end], as[nb+2:end]) : (Float64[], Float64[])
    end

    "Learn θ̄ and Σ̂ from the samples after burn-in."
    function learn!()
        kk, aa = kept()
        if length(kk) < 30
            status.text[] = "Need at least 30 samples after the burn-in to learn the covariance Σ (run more, or lower the burn-in)."
            status.color = :red3; return
        end
        mk, ma = mean_(kk), mean_(aa)
        C = Sym2(var_(kk), sum((kk .- mk) .* (aa .- ma)) / (length(kk) - 1), var_(aa))
        if det2(C) <= 0
            status.text[] = "The samples are degenerate (all equal?) – cannot learn Σ."; status.color = :red3; return
        end
        learned[] = ((mk, ma), C, true)
        draw_chain!()
        status.text[] = @sprintf("Learned from %d samples: mean = (%.3g, %.3g),  sd = (%.3g, %.3g),  ρ = %.2f  →  Σ used by proposals 3 and 4.",
                                 length(kk), mk, ma, sd2(C)..., corr2(C))
        status.color = :gray20
    end

    "Proposal density at the current state (ellipses 1σ, 2σ; for proposal 2 the next 1-D step ±2s)."
    function draw_q!()
        Q = sampler(); θ = (ks[end], as[end])
        if Q.type == 2
            s = Q.s
            o_qell[] = isodd(nit() + 1) ? [Point2f(θ[1] - 2s, θ[2]), Point2f(θ[1] + 2s, θ[2])] :
                                          [Point2f(θ[1], θ[2] - 2s), Point2f(θ[1], θ[2] + 2s)]
        else
            c, C = qpars(Q, θ)
            o_qell[] = vcat(ellipse(c, C, 1.0), ellipse(c, C, 2.0))
        end
    end

    function draw_chain!()
        P = prior(); Sd = S[]
        n  = nit(); kc, ac = ks[end], as[end]
        line(k, a) = [Point2f(XLIM[1], a + k * XLIM[1]), Point2f(XLIM[2], a + k * XLIM[2])]
        # data panel
        o_cur[] = line(kc, ac)
        i0 = max(1, length(ks) - NSPAG + 1)
        o_spag[] = reduce(vcat, [line(ks[i], as[i]) for i in i0:length(ks)])
        L = laststep[]; show1 = lastn[] == 1 && L !== nothing
        vis_prop[] = show1
        if show1
            c_prop[]  = L.acc ? :forestgreen : C_REJ
            o_prop[]  = line(L.θp...)
            o_mprop[] = [Point2f(L.θp...)]
            o_arrow[] = [Point2f(L.θ_old...), Point2f(L.θp...)]
        end
        # misfit map
        o_vis[]  = Point2f.(ks, as)
        o_path[] = Point2f.(ks[max(1, end - NPATH):end], as[max(1, end - NPATH):end])
        rej = props[.!accs]
        o_rej[]  = isempty(rej) ? PN : [Point2f(p...) for p in rej]
        o_mcur[] = [Point2f(kc, ac)]
        draw_q!()
        # marginals
        kk, aa = kept()
        ck, dk, wk = density_hist(kk); ca, da, wa = density_hist(aa)
        o_hk[] = Point2f.(ck, dk); o_bwk[] = wk
        o_ha[] = Point2f.(ca, da); o_bwa[] = wa
        vis_hist[] = !isempty(kk)
        # traces
        o_trk[] = Point2f.(0:n, ks); o_tra[] = Point2f.(0:n, as)
        draw_limits!()
        # text
        m, C = exact_posterior(Sd, P); sk, sa = sd2(C)
        θh, _ = lsq(Sd)
        nacc = count(accs)
        txt = @sprintf("iterations: %d    accepted: %d (%.0f %%)    current (k, a) = (%s, %s),  χ² = %.4g",
                       n, nacc, n == 0 ? 0.0 : 100nacc / n, fmt(kc), fmt(ac), chi2(kc, ac, Sd))
        if show1
            txt *= @sprintf("\nlast step: (%s, %s) → θ′ = (%s, %s)    χ²: %.4g → %.4g", fmt(L.θ_old[1]), fmt(L.θ_old[2]),
                            fmt(L.θp[1]), fmt(L.θp[2]), chi2(L.θ_old..., Sd), chi2(L.θp..., Sd))
            txt *= "\n   p(θ′|d)/p(θ|d) = " * fmtr(L.A) * "   q(θ|θ′)/q(θ′|θ) = " * fmtr(L.H) *
                   (tg_hast.active[] ? "" : " (ignored)") * "   α = " * fmtr(L.α) * @sprintf(",  u = %.3g", L.u) *
                   (L.acc ? "  → ACCEPTED" : "  → REJECTED")
        else
            txt *= "\n\n"
        end
        if isempty(kk)
            txt *= "\nsamples after burn-in: none yet\n"
        else
            nk = length(kk)
            ρ  = nk > 2 ? sum((kk .- mean_(kk)) .* (aa .- mean_(aa))) / ((nk - 1) * sqrt(var_(kk) * var_(aa))) : NaN
            txt *= @sprintf("\nsamples after burn-in: %d    k = %s ± %s    a = %s ± %s    ρ = %.2f", nk,
                            fmt2(mean_(kk)), fmt2(nk > 1 ? sqrt(var_(kk)) : 0.0), fmt2(mean_(aa)),
                            fmt2(nk > 1 ? sqrt(var_(aa)) : 0.0), ρ)
            ek, ea = ess(kk), ess(aa)
            txt *= isnan(ek) ? "\n   effective sample size: (needs ≥ 50 samples)" :
                   @sprintf("\n   effective sample size ≈ %.0f (k), %.0f (a)   –  %.1f %% of the samples", ek, ea,
                            100 * min(ek, ea) / nk)
        end
        txt *= @sprintf("\nexact posterior: k = %s ± %s    a = %s ± %s    ρ = %.2f    (least squares: %s, %s)",
                        fmt2(m[1]), fmt2(sk), fmt2(m[2]), fmt2(sa), corr2(C), fmt2(θh[1]), fmt2(θh[2]))
        info.text[] = txt
        axM.title = "misfit χ²(k, a)  –  click to set the start"
        axK.title = isempty(kk) ? "marginal posteriors  (no samples yet)" : "marginal posteriors  ($(length(kk)) samples)"
        axTk.title = "trace plots   –   " * (tg_hast.active[] || ptype[] in (1, 2, 3) ? "Metropolis–Hastings" :
                                             "NO Hastings correction (biased for this proposal)")
        axTk.titlecolor = tg_hast.active[] || ptype[] in (1, 2, 3) ? :black : C_REJ
    end

    "Axis limits of the marginals and traces (the map limits are set by draw_map!)."
    function draw_limits!()
        k1, k2, a1, a2 = window()
        P = prior(); m, C = exact_posterior(S[], P); sk, sa = sd2(C)
        yk = max(maximum(p[2] for p in o_pk[] if k1 <= p[1] <= k2; init = 0.0), tg_exact.active[] ? gauss(0, 0, sk) : 0.0)
        ya = max(maximum(p[1] for p in o_pa[] if a1 <= p[2] <= a2; init = 0.0), tg_exact.active[] ? gauss(0, 0, sa) : 0.0)
        hk = vis_hist[] ? maximum(p[2] for p in o_hk[]) : 0.0
        ha = vis_hist[] ? maximum(p[2] for p in o_ha[]) : 0.0
        limits!(axK, k1, k2, 0, 1.12 * max(yk, min(hk, 2.5yk)) + 1e-9)
        limits!(axA, 0, 1.12 * max(ya, min(ha, 2.5ya)) + 1e-9, a1, a2)
        n = nit(); xr = max(n, 10) * 1.02
        limits!(axTk, -0.02 * max(n, 10), xr, k1, k2)
        limits!(axTa, -0.02 * max(n, 10), xr, a1, a2)
        b = min(sl_burn.value[], n)
        o_burnk[] = Rect2f(0, k1, b, k2 - k1); o_burna[] = Rect2f(0, a1, b, a2 - a1)
        o_bandk[] = Rect2f(0, m[1] - sk, xr, 2sk); o_banda[] = Rect2f(0, m[2] - sa, xr, 2sa)
    end

    data_pending() = (sl_kt.value[], sl_at.value[], sl_sig.value[], sl_n.value[]) != used[]
    function set_status!(msg; warn = false)
        status.text[] = msg
        status.color = warn ? :red3 : :gray20
    end

    function initialise!()
        init_data!(); reset_chain!(); draw_map!(); draw_prior!(); draw_chain!()
        set_status!("Data initialised and χ²(k, a) computed. Choose the priors and the proposal, then press \"Run MCMC batch\".")
    end
    function restart!(why)
        reset_chain!(); draw_prior!(); draw_chain!()
        set_status!(why * @sprintf(" → chain restarted at (k₀, a₀) = (%.2f, %.2f).", start[]...))
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
    for sl in (sl_mk, sl_sk, sl_ma, sl_sa)
        on(_ -> restart!("prior changed (new target posterior)"), sl.value)
    end
    on(_ -> restart!("chain reset"), btn_reset.clicks)
    on(btn_run.clicks) do _
        run_batch!(sl_batch.value[]); draw_chain!()
        data_pending() && set_status!("Note: the data sliders were changed but not applied (press \"Initialise data\").",
                                      warn = true)
    end
    on(_ -> learn!(), btn_learn.clicks)
    on(menu.selection) do sel
        sel === nothing && return
        t = findfirst(==(sel), PROPOSALS)
        (t === nothing || t == ptype[]) && return
        ptype[] = t
        draw_chain!()
        set_status!("Proposal: " * PROPOSALS[t] * "." *
                    (t in (3, 4) && !learned[][3] ? "  Σ = I and mean = start until you press \"learn mean, Σ from chain\"." :
                     t == 4 ? "  Use s ≥ 1.5: q must be wider than the posterior, or the chain gets stuck." :
                     "  The target is unchanged, so the chain continues."),
                    warn = t in (3, 4) && !learned[][3])
    end
    on(_ -> draw_q!(), sl_step.value)
    on(tg_hast.active) do h
        draw_chain!()
        set_status!(h ? "Hastings correction ON." :
                        "Hastings correction OFF: α = min(1, posterior ratio). Proposals 4 and 5 now sample the wrong " *
                        "distribution (press \"Reset chain\" and run).", warn = !h)
    end
    on(_ -> draw_chain!(), sl_burn.value)
    on(_ -> (draw_map!(); draw_prior!(); draw_chain!()), tg_zoom.active)
    on(_ -> draw_limits!(), tg_exact.active)
    on(_ -> draw_limits!(), tg_prior.active)
    register_interaction!(axM, :pick_start) do event::MouseEvent, _
        if event.type === MouseEventTypes.leftclick
            start[] = (clamp(event.data[1], KLIM...), clamp(event.data[2], ALIM...))
            restart!("starting point changed")
        end
        return Consume(false)
    end

    # ------------------------------------------------------- initialisation
    rng_d[] = Xoshiro(SEED)
    xall[]  = XLIM[1] .+ (XLIM[2] - XLIM[1]) .* rand(rng_d[], NMAX)
    eall[]  = randn(rng_d[], NMAX)
    initialise!()
    return fig, (init = initialise!, noise = () -> (new_noise!(); initialise!()),
                 run = n -> (run_batch!(n); draw_chain!()), reset = () -> restart!("chain reset"),
                 learn = learn!, proposal = t -> (menu.i_selected[] = t),
                 setstart = θ -> (start[] = θ; restart!("starting point changed")),
                 sliders = (kt = sl_kt, at = sl_at, sig = sl_sig, n = sl_n, mk = sl_mk, sk = sl_sk, ma = sl_ma, sa = sl_sa,
                            step = sl_step, batch = sl_batch, burn = sl_burn),
                 toggles = (zoom = tg_zoom, exact = tg_exact, prior = tg_prior, rej = tg_rej, q = tg_q, path = tg_path,
                            hastings = tg_hast),
                 ks = ks, as = as)
end

# =============================================================================
# RUN
# =============================================================================
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    fig, app = build_app()
    screen = display(fig)
    isinteractive() || wait(screen)     # keep the window open when run from a terminal
end
