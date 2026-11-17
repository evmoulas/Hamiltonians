#=
Introduction_HMC_2 – Hamiltonian Monte Carlo for a 2-parameter fit  y = a + k·x
================================================================================

Same problem as Introduction_MCMC_3: data yᵢ = a_true + k_true·xᵢ + σ·εᵢ,
model y = a + k·x, Gaussian priors on k and a, θ = (k, a).

  potential   U(θ) = −ln p(θ|d) = ½χ²(k, a) + ½(k−μ_k)²/σ_k² + ½(a−μ_a)²/σ_a²
  momentum    p = (p_k, p_a) ~ N(0, M)            (M = 2×2 MASS MATRIX)
  Hamiltonian H(θ, p) = U(θ) + ½ pᵀ M⁻¹ p

One HMC iteration: draw p₀ ~ N(0, M); L leapfrog steps of size ε
     p ← p − ½ε∇U(θ);   θ ← θ + ε M⁻¹p;   p ← p − ½ε∇U(θ);
accept the end point with α = min(1, e^(−ΔH)).

New in two dimensions – the shape of the posterior:
  near its maximum U ≈ ½(θ − m)ᵀ Λ (θ − m) (Λ = posterior precision). The particle
  oscillates with the angular frequencies ω = √(eigenvalues of M⁻¹Λ).
  * The leapfrog scheme is stable only for ε < 2/ω_max (the STIFF direction).
  * Exploring the long direction of the valley takes about π/ω_min.
  * A trajectory of a QUARTER period (L·ε ≈ π/(2ω)) gives an almost independent
    sample; HALF a period only reflects θ → 2m − θ (the distance from the mean
    hardly changes); a WHOLE period brings the particle back (U-turn).
  With M = I and a narrow, tilted posterior (here ω_max/ω_min ≈ 8) ε is limited by
  the stiff direction, while the long direction needs a long trajectory – and for
  a fixed L·ε the stiff oscillation may be close to a multiple of half a period
  (RESONANCE: the chain does not move across the valley). Remedies:
    – jitter ε randomly from trajectory to trajectory (toggle "jitter ε");
    – a mass matrix with M⁻¹ ≈ posterior covariance Σ: all ω become equal (= 1),
      the posterior looks round to the particle, ε can be large and
      L·ε ≈ π/2 gives nearly independent samples. This is the 2-D analogue of the
      learned proposal covariance in Introduction_MCMC_3 ("mass matrix
      adaptation", used by Stan and other HMC codes).

Mass matrix (menu):  identity  M = I  |  diagonal  M = diag(1/var_k, 1/var_a)  |
full  M = Σ⁻¹, with var and Σ learned from the chain (button "learn Σ from chain").

Panels
  upper left   : data, true line, current fit
  upper right  : misfit map χ²(k, a) with the chain and the leapfrog trajectories
                 in parameter space; two projections of the 4-D phase space:
                 (k, p_k) and (a, p_a) with trajectories and momentum refreshes
  lower right  : marginal posteriors of k and a, trace plots
  lower left   : controls; click in the χ² map to set the starting point

What to try
  * M = I, ε = 0.05, L = 20 (defaults): works, ESS about half the number of samples.
  * M = I, ε = 0.15, L = 20: almost all accepted and the ESS estimate looks fine,
    but the stiff oscillation is close to a multiple of half a period (RESONANCE,
    the note under the buttons warns). The chain hardly moves across the valley
    and the result changes from run to run: in 12 runs of 2000 samples the sd of
    k was between 0.13 and 0.28 (exact 0.156), ρ between −0.96 and −0.41. With
    "jitter ε" on: sd 0.150 – 0.162, ρ −0.80 – −0.74. (Press Reset + run a few times.)
    From the default start (−0.5, −5) with these settings the chain does not even
    burn in: after 1000 iterations it still swings across the whole k range
    (H ≈ 400) – the momentum refreshes cannot remove the energy of a reflected
    stiff oscillation. Jitter ε, or a smaller ε, removes the problem.
  * M = I, ε = 0.2: above the stability limit 2/ω_max ≈ 0.197 – everything diverges.
  * pilot run, "learn Σ", M = full, ε = 0.3, L = 5 (L·ε ≈ π/2, a quarter period):
    nearly independent samples. With L = 10 (half a period) and a start far away
    the chain only reflects between two far points and never burns in.
  * diagonal M: corrects the different scales of k and a but not the correlation.

Run from a terminal:      julia Introduction_HMC_2.jl
or from the Julia REPL:   include("Introduction_HMC_2.jl")
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
const EPS      = 0.05               # default leapfrog step ε
const NLEAP    = 20                 # default number of leapfrog steps L
const BATCHES  = [1, 10, 100, 1000] # batch sizes of the "Run" button
const NSPAG    = 40                 # number of last chain states drawn in the data panel
const NTRAJ    = 20                 # number of last trajectories drawn
const DCHI     = [2.30, 6.18, 11.8] # Δχ² of the 68 / 95 / 99.7 % regions (2 parameters)
const DIVERGE  = 1000.0             # ΔH above this: divergent trajectory

const MASSES = ["identity   M = I", "diagonal   M = diag(1/var)", "full   M = Σ⁻¹"]

const C_DATA  = :black
const C_TRUE  = :gray40
const C_CUR   = :darkorange2        # current state
const C_CHAIN = :seagreen           # visited states / histograms / accepted trajectories
const C_REJ   = :red3               # rejected
const C_PRIOR = :royalblue3
const C_POST  = :black              # exact posterior
const C_MIS   = :purple4            # misfit contours
const C_KICK  = :gray30             # momentum refresh

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
"Eigenvalues of the product A·B of two symmetric positive definite 2×2 matrices (real, > 0)."
function eig_prod(A::Sym2, B::Sym2)
    m11 = A.xx * B.xx + A.xy * B.xy; m12 = A.xx * B.xy + A.xy * B.yy
    m21 = A.xy * B.xx + A.yy * B.xy; m22 = A.xy * B.xy + A.yy * B.yy
    t = m11 + m22; d = m11 * m22 - m12 * m21
    r = sqrt(max(t^2 / 4 - d, 0.0))
    return t / 2 - r, t / 2 + r
end

"Sums of the data: N, Σx, Σx², Σy, Σxy, Σy² and σ²."
struct DataSums
    n::Float64; x::Float64; xx::Float64; y::Float64; xy::Float64; yy::Float64; s2::Float64
end
DataSums(x, y, σ) = DataSums(length(x), sum(x), sum(abs2, x), sum(y), sum(x .* y), sum(abs2, y), σ^2)

"Misfit χ²(k, a) = Σ (yᵢ − a − k xᵢ)² / σ²  (expanded: O(1))."
chi2(k, a, S::DataSums) =
    max(S.yy - 2a * S.y - 2k * S.xy + S.n * a^2 + 2a * k * S.x + k^2 * S.xx, 0.0) / S.s2

struct Prior
    mk::Float64; sk::Float64; ma::Float64; sa::Float64
end
logprior(k, a, P::Prior) = -0.5 * ((k - P.mk) / P.sk)^2 - 0.5 * ((a - P.ma) / P.sa)^2
logpost(k, a, S, P) = -0.5 * chi2(k, a, S) + logprior(k, a, P)
gradlogpost(k, a, S, P) = ((S.xy - a * S.x - k * S.xx) / S.s2 - (k - P.mk) / P.sk^2,
                           (S.y - S.n * a - k * S.x) / S.s2 - (a - P.ma) / P.sa^2)

"Exact Gaussian posterior: mean (k, a) and covariance."
function exact_posterior(S::DataSums, P::Prior)
    Λ = Sym2(S.xx / S.s2 + 1 / P.sk^2, S.x / S.s2, S.n / S.s2 + 1 / P.sa^2)
    C = inv2(Λ)
    return mul(C, (S.xy / S.s2 + P.mk / P.sk^2, S.y / S.s2 + P.ma / P.sa^2)), C
end
function lsq(S::DataSums)
    C = inv2(Sym2(S.xx / S.s2, S.x / S.s2, S.n / S.s2))
    return mul(C, (S.xy / S.s2, S.y / S.s2)), C
end

gauss(x, m, s) = exp(-0.5 * ((x - m) / s)^2) / (s * sqrt(2π))

"Points on the ellipse (θ − m)ᵀ C⁻¹ (θ − m) = r² (NaN at the end separates curves)."
function ellipse(m, C::Sym2, r; n = 120)
    l11, l21, l22 = chol2(C)
    pts = [Point2f(m[1] + r * l11 * cos(t), m[2] + r * (l21 * cos(t) + l22 * sin(t))) for t in range(0, 2π, length = n)]
    push!(pts, Point2f(NaN, NaN))
    return pts
end

"HMC settings: data, prior, ε, L, mass matrix M and its inverse."
struct HMC
    S::DataSums
    P::Prior
    ε::Float64
    L::Int
    M::Sym2
    Minv::Sym2
end
HMC(S, P, ε, L, M) = HMC(S, P, ε, L, M, inv2(M))

U(θ, Q::HMC) = -logpost(θ[1], θ[2], Q.S, Q.P)
dU(θ, Q::HMC) = (g = gradlogpost(θ[1], θ[2], Q.S, Q.P); (-g[1], -g[2]))
H(θ, p, Q::HMC) = U(θ, Q) + 0.5 * quad(Q.Minv, p)

"""
L leapfrog steps from (θ, p). Returns the phase-space points (k, a, p_k, p_a) after
every step (momentum synchronised with the position) and the end point.
"""
function leapfrog(θ, p, Q::HMC)
    pts = [(θ..., p...)]
    g   = dU(θ, Q)
    ph  = (p[1] - 0.5Q.ε * g[1], p[2] - 0.5Q.ε * g[2])
    for i in 1:Q.L
        v  = mul(Q.Minv, ph)
        θ  = (θ[1] + Q.ε * v[1], θ[2] + Q.ε * v[2])
        g  = dU(θ, Q)
        push!(pts, (θ..., ph[1] - 0.5Q.ε * g[1], ph[2] - 0.5Q.ε * g[2]))
        (!all(isfinite, θ) || maximum(abs, θ) > 1e6) && return pts, θ, (NaN, NaN)
        i < Q.L && (ph = (ph[1] - Q.ε * g[1], ph[2] - Q.ε * g[2]))
    end
    return pts, θ, (ph[1] - 0.5Q.ε * g[1], ph[2] - 0.5Q.ε * g[2])
end

"One HMC iteration from θ."
function hmc_step(rng, θ, Q::HMC)
    l11, l21, l22 = chol2(Q.M)
    z1, z2 = randn(rng), randn(rng)
    p0 = (l11 * z1, l21 * z1 + l22 * z2)                 # p₀ ~ N(0, M)
    pts, θ1, p1 = leapfrog(θ, p0, Q)
    H0 = H(θ, p0, Q)
    H1 = all(isfinite, θ1) && all(isfinite, p1) ? H(θ1, p1, Q) : Inf
    ΔH = H1 - H0
    div = !(ΔH < DIVERGE)
    α  = div ? 0.0 : exp(min(0.0, -ΔH))
    u  = rand(rng)
    return (θ_old = θ, p0 = p0, θp = θ1, pp = p1, pts = pts, H0 = H0, H1 = H1, ΔH = ΔH, α = α, u = u,
            acc = !div && u < α, div = div)
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
    rng_d  = Ref(Xoshiro(SEED))
    rng_c  = Xoshiro(SEED + 1000)
    xall   = Ref(zeros(NMAX)); eall = Ref(zeros(NMAX))
    xd     = Ref(Float64[]);   yd   = Ref(Float64[])
    S      = Ref(DataSums(2, 1, 1, 0, 0, 0, 1))
    used   = Ref((K_TRUE, A_TRUE, SIGMA, NDATA))
    start  = Ref(START)
    ks     = Float64[]; as = Float64[]                # visited states (index 1 = start)
    props  = NTuple{2,Float64}[]; accs = Bool[]
    ndiv   = Ref(0)
    recent = Any[]                                    # the last NTRAJ iterations
    lastn  = Ref(0)
    mtype  = Ref(1)                                   # mass matrix type
    learned = Ref((I2, false))                        # (Σ learned from the chain, learned?)

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Introduction to HMC 2  –  Hamiltonian Monte Carlo for two parameters  (Bayesian fit of  y = a + k·x)",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(540))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")        # no fixed height: it takes the space the controls leave

    top = right[1, 1] = GridLayout()
    axM  = Axis(top[1:2, 1]; xlabel = "slope k", ylabel = "intercept a")
    axPk = Axis(top[1, 2]; xlabel = "k", ylabel = "p_k", title = "phase space (k, p_k)")
    axPa = Axis(top[2, 2]; xlabel = "a", ylabel = "p_a", title = "phase space (a, p_a)")
    colsize!(top, 2, Relative(0.36)); colgap!(top, 12); rowgap!(top, 6)
    mid = right[2, 1] = GridLayout()
    axK = Axis(mid[1, 1]; xlabel = "k", ylabel = "p(k|d)")
    axA = Axis(mid[1, 2]; xlabel = "a", ylabel = "p(a|d)")
    bot = right[3, 1] = GridLayout()
    axTk = Axis(bot[1, 1]; xlabel = "iteration", ylabel = "k", title = "trace of k")
    axTa = Axis(bot[1, 2]; xlabel = "iteration", ylabel = "a", title = "trace of a")
    for g in (mid, bot)
        colgap!(g, 16)
    end
    rowsize!(right, 1, Auto(2.3)); rowgap!(right, 6)
    for a in (axD, axM, axPk, axPa, axK, axA, axTk, axTa), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
    end

    # display options (below the trace plots)
    tg = right[4, 1] = GridLayout(halign = :center, tellwidth = false)
    tg_zoom  = Toggle(tg[1, 1], active = false); Label(tg[1, 2], "zoom on posterior", halign = :left)
    tg_exact = Toggle(tg[1, 3], active = true);  Label(tg[1, 4], "exact posterior", halign = :left)
    tg_prior = Toggle(tg[1, 5], active = true);  Label(tg[1, 6], "prior", halign = :left)
    tg_rej   = Toggle(tg[2, 1], active = true);  Label(tg[2, 2], "rejected trajectories", halign = :left)
    tg_old   = Toggle(tg[2, 3], active = true);  Label(tg[2, 4], "older trajectories", halign = :left)
    tg_vis   = Toggle(tg[2, 5], active = true);  Label(tg[2, 6], "visited states", halign = :left)
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
    btn_init  = Button(bD[1, 1], label = "Initialise data (compute χ² map)", buttoncolor = RGBf(0.80, 0.88, 1.0), height = 28)
    btn_noise = Button(bD[1, 2], label = "new noise realisation", height = 28)

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
    Label(left[8, 1], "3 · HMC   H = U(θ) + ½pᵀM⁻¹p,   α = min(1, e^(−ΔH))", font = :bold, halign = :left, tellwidth = false)
    mg = left[9, 1] = GridLayout(halign = :left)
    Label(mg[1, 1], "mass M:", halign = :right)
    menu = Menu(mg[1, 2], options = MASSES, default = MASSES[1], width = 215, height = 28)
    tg_jit = Toggle(mg[1, 3], active = false)
    Label(mg[1, 4], "jitter ε\n(× U(0.8, 1.2))", halign = :left, justification = :left, fontsize = 11)
    colgap!(mg, 6); colgap!(mg, 2, 16)
    sgS = SliderGrid(left[10, 1],
        (label = "leapfrog step ε", range = filter(v -> 0.001 <= v <= 3, nice), startvalue = EPS, format = "{:.3g}"),
        (label = "leapfrog steps L", range = 1:100, startvalue = NLEAP, format = "{:d}"),
        (label = "batch size", range = BATCHES, startvalue = BATCHES[1], format = "{:d} samples"),
        (label = "burn-in", range = 0:10:5000, startvalue = 0, format = "discard first {:d}");
        tellheight = true, width = 500, halign = :left)
    sl_eps, sl_L, sl_batch, sl_burn = sgS.sliders
    bS = left[11, 1] = GridLayout(halign = :left)
    btn_run   = Button(bS[1, 1], label = "Run HMC batch", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 170, height = 28)
    btn_reset = Button(bS[1, 2], label = "Reset chain", height = 28)
    btn_learn = Button(bS[1, 3], label = "learn Σ from chain", buttoncolor = RGBf(1.0, 0.90, 0.78), height = 28)
    tlen = Label(left[12, 1], " ", halign = :left, tellwidth = false, justification = :left, color = :gray35, fontsize = 11)

    # fixed heights (6 and 2 lines), so that the data axis does not jump when the text changes
    info   = Label(left[13, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   fontsize = 11.5, height = 86)
    status = Label(left[14, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 34)
    rowgap!(left, 4)
    for sg in (sgD, sgP, sgS)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    P0 = Point2f[(0, 0)]; PN = Point2f[(NaN, NaN)]; PN2 = Point2f[(NaN, NaN), (NaN, NaN)]
    # data panel
    o_dat  = Observable(P0); o_err = Observable(Point2f[(0, 0), (0, 0)])
    o_true = Observable(Point2f[(0, 0), (1, 0)]); o_spag = Observable(Point2f[(0, 0), (0, 0)])
    o_cur  = Observable(Point2f[(0, 0), (1, 0)])
    o_prop = Observable(Point2f[(0, 0), (1, 0)]); c_last = Observable{Any}(C_CHAIN); vis_prop = Observable(false)
    # misfit map
    o_gx = Observable(collect(range(0, 1, length = NG))); o_gy = Observable(collect(range(0, 1, length = NG)))
    o_gz = Observable(zeros(NG, NG)); o_gl = Observable(zeros(NG, NG))
    o_lev1 = Observable(collect(1.0:3)); o_lev2 = Observable(collect(1.0:10))
    o_true2 = Observable(P0); o_lsq = Observable(P0)
    o_pell = Observable(PN); o_xell = Observable(PN)
    o_vis  = Observable(P0); o_rej = Observable(PN)
    o_mcur = Observable(P0)
    # trajectories: in the map (k, a) and in the two phase planes; index 1 = map, 2 = (k, p_k), 3 = (a, p_a)
    o_tacc  = [Observable(PN) for _ in 1:3]; o_trej = [Observable(PN) for _ in 1:3]
    o_tlast = [Observable(PN) for _ in 1:3]; o_tst  = [Observable(PN) for _ in 1:3]; o_tend = [Observable(PN) for _ in 1:3]
    o_kick  = [Observable(PN2) for _ in 1:2]; o_pcur = [Observable(PN) for _ in 1:2]
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
    lines!(axD, o_prop; color = c_last, linewidth = 2, linestyle = :dot, visible = vis_prop, label = "proposal θ′")
    lines!(axD, o_cur; color = C_CUR, linewidth = 3, label = "current fit  y = a + k·x")
    linesegments!(axD, o_err; color = (C_DATA, 0.6), linewidth = 1.2)
    scatter!(axD, o_dat; color = C_DATA, markersize = 9, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 11, rowgap = 0, colgap = 6, nbanks = 2, patchsize = (16, 10),
               padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

    # misfit map with trajectories
    heatmap!(axM, o_gx, o_gy, o_gl; colormap = :Blues, alpha = 0.75)
    contour!(axM, o_gx, o_gy, o_gz; levels = o_lev2, color = (:gray40, 0.45), linewidth = 0.7)
    contour!(axM, o_gx, o_gy, o_gz; levels = o_lev1, color = C_MIS, linewidth = 1.8, label = "Δχ² = 2.3, 6.2, 11.8")
    lines!(axM, o_pell; color = (C_PRIOR, 0.8), linewidth = 1.5, linestyle = :dash, visible = tg_prior.active,
           label = "prior 1σ, 2σ")
    lines!(axM, o_xell; color = C_POST, linewidth = 1.8, linestyle = :dash, visible = tg_exact.active,
           label = "exact posterior 1σ, 2σ")
    scatter!(axM, o_vis; color = (C_CHAIN, 0.35), markersize = 5, visible = tg_vis.active, label = "visited states")
    scatter!(axM, o_rej; color = (C_REJ, 0.35), marker = :xcross, markersize = 7, visible = tg_rej.active)
    scatter!(axM, o_true2; color = :white, marker = :star5, markersize = 18, strokecolor = :black, strokewidth = 1.2,
             label = "true (k, a)")
    scatter!(axM, o_lsq; color = C_MIS, marker = :cross, markersize = 14, label = "least squares")

    # trajectories (map and phase planes)
    for (j, ax) in enumerate((axM, axPk, axPa))
        vis_rej = lift((a, b) -> a && b, tg_rej.active, tg_old.active)
        lines!(ax, o_trej[j]; color = (C_REJ, 0.35), linewidth = 1, linestyle = :dash, visible = vis_rej)
        lines!(ax, o_tacc[j]; color = (C_CHAIN, 0.5), linewidth = 1.2, visible = tg_old.active)
        j > 1 && linesegments!(ax, o_kick[j-1]; color = (C_KICK, 0.6), linewidth = 1, linestyle = :dot,
                               visible = tg_old.active)
        scatterlines!(ax, o_tlast[j]; color = c_last, markersize = 4, linewidth = 2, label = j == 1 ? "last trajectory" : nothing)
        scatter!(ax, o_tst[j]; color = :white, strokecolor = c_last, strokewidth = 2, markersize = 10)
        scatter!(ax, o_tend[j]; color = c_last, marker = :diamond, markersize = 12, strokecolor = :black, strokewidth = 1)
        j > 1 && scatter!(ax, o_pcur[j-1]; color = C_CUR, markersize = 13, strokecolor = :black, strokewidth = 1.5)
    end
    scatter!(axM, o_mcur; color = C_CUR, markersize = 15, strokecolor = :black, strokewidth = 1.5, label = "current state")
    axislegend(axM; position = :rt, labelsize = 9, rowgap = -2, padding = (4, 4, 2, 2), patchsize = (12, 8),
               backgroundcolor = (:white, 0.8))

    # marginals
    for (ax, h, bw, pr, ex, tv) in ((axK, o_hk, o_bwk, o_pk, o_ek, o_kt), (axA, o_ha, o_bwa, o_pa, o_ea, o_at))
        barplot!(ax, h; width = bw, gap = 0, color = (C_CHAIN, 0.45), strokecolor = (C_CHAIN, 0.9),
                 strokewidth = 0.5, visible = vis_hist, label = "samples")
        lines!(ax, pr; color = C_PRIOR, linewidth = 2, visible = tg_prior.active, label = "prior")
        lines!(ax, ex; color = C_POST, linewidth = 1.8, linestyle = :dash, visible = tg_exact.active, label = "exact")
        vlines!(ax, tv; color = C_TRUE, linestyle = :dash, linewidth = 1.2)
    end
    axislegend(axK; position = :rt, labelsize = 10, rowgap = -2, padding = (4, 4, 2, 2), patchsize = (12, 8),
               backgroundcolor = (:white, 0.8))

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
    "Mass matrix of the chosen type (identity until Σ has been learned)."
    function massmatrix()
        Σ, ok = learned[]
        (mtype[] == 1 || !ok) && return I2
        mtype[] == 2 && return Sym2(1 / Σ.xx, 0.0, 1 / Σ.yy)
        return inv2(Σ)
    end
    hmc() = HMC(S[], prior(), sl_eps.value[], sl_L.value[], massmatrix())
    nit() = length(ks) - 1

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

    function draw_prior!()
        P = prior(); m, C = exact_posterior(S[], P)
        Cp = Sym2(P.sk^2, 0.0, P.sa^2)
        o_pell[] = vcat(ellipse((P.mk, P.ma), Cp, 1.0), ellipse((P.mk, P.ma), Cp, 2.0))
        o_xell[] = vcat(ellipse(m, C, 1.0), ellipse(m, C, 2.0))
        k1, k2, a1, a2 = window(); sk, sa = sd2(C)
        kk = sort(vcat(range(min(k1, KLIM[1]), max(k2, KLIM[2]), length = 2001), range(m[1] - 6sk, m[1] + 6sk, length = 601)))
        aa = sort(vcat(range(min(a1, ALIM[1]), max(a2, ALIM[2]), length = 2001), range(m[2] - 6sa, m[2] + 6sa, length = 601)))
        o_pk[] = Point2f.(kk, gauss.(kk, P.mk, P.sk)); o_ek[] = Point2f.(kk, gauss.(kk, m[1], sk))
        o_pa[] = Point2f.(aa, gauss.(aa, P.ma, P.sa)); o_ea[] = Point2f.(aa, gauss.(aa, m[2], sa))
        o_mk[] = m[1]; o_ma[] = m[2]
    end

    function reset_chain!()
        empty!(ks); empty!(as); empty!(props); empty!(accs); empty!(recent)
        θ0 = start[]
        push!(ks, θ0[1]); push!(as, θ0[2])
        ndiv[] = 0; lastn[] = 0
        startlabel.text[] = @sprintf("start (k₀, a₀) = (%.2f, %.2f)  –  click in the χ² map to change it", θ0...)
    end

    function run_batch!(n)
        Q = hmc()
        θ = (ks[end], as[end])
        for _ in 1:n
            Qi = tg_jit.active[] ? HMC(Q.S, Q.P, Q.ε * (0.8 + 0.4rand(rng_c)), Q.L, Q.M, Q.Minv) : Q
            r = hmc_step(rng_c, θ, Qi)
            push!(props, r.θp); push!(accs, r.acc)
            r.div && (ndiv[] += 1)
            r.acc && (θ = r.θp)
            push!(ks, θ[1]); push!(as, θ[2])
            push!(recent, r)
            length(recent) > NTRAJ && popfirst!(recent)
        end
        lastn[] = n
    end

    function kept()
        nb = sl_burn.value[]
        nit() > nb ? (ks[nb+2:end], as[nb+2:end]) : (Float64[], Float64[])
    end

    "Learn the posterior covariance Σ from the samples after burn-in (for the mass matrix)."
    function learn!()
        kk, aa = kept()
        if length(kk) < 30
            set_status!("Need at least 30 samples after the burn-in to learn Σ (run more, or lower the burn-in).", warn = true)
            return
        end
        mk, ma = mean_(kk), mean_(aa)
        C = Sym2(var_(kk), sum((kk .- mk) .* (aa .- ma)) / (length(kk) - 1), var_(aa))
        if det2(C) <= 0
            set_status!("The samples are degenerate (all equal?) – cannot learn Σ.", warn = true); return
        end
        learned[] = (C, true)
        draw_chain!()
        set_status!(@sprintf("Learned Σ from %d samples: sd = (%.3g, %.3g), ρ = %.2f  →  used by the diagonal and full mass matrix.",
                             length(kk), sd2(C)..., corr2(C)))
    end

    "State in phase space after iteration r (end point if accepted, start point otherwise)."
    state_after(r) = r.acc ? (r.θp..., r.pp...) : (r.θ_old..., r.p0...)
    "Projection of a phase-space point on panel j (1: (k, a), 2: (k, p_k), 3: (a, p_a))."
    proj(q, j) = j == 1 ? Point2f(q[1], q[2]) : j == 2 ? Point2f(q[1], q[3]) : Point2f(q[2], q[4])

    function draw_traj!()
        nr = length(recent)
        for j in 1:3
            acc = Point2f[]; rej = Point2f[]
            for i in 1:nr-1
                r = recent[i]
                pts = [proj(q, j) for q in r.pts if all(isfinite, q)]
                append!(r.acc ? acc : rej, pts); push!(r.acc ? acc : rej, Point2f(NaN, NaN))
            end
            o_tacc[j][] = isempty(acc) ? PN : acc
            o_trej[j][] = isempty(rej) ? PN : rej
            if j > 1
                kick = Point2f[]
                for i in 2:nr
                    s = state_after(recent[i-1]); p0 = recent[i].p0
                    push!(kick, proj(s, j), proj((s[1], s[2], p0[1], p0[2]), j))
                end
                o_kick[j-1][] = isempty(kick) ? PN2 : kick
            end
            if nr == 0
                o_tlast[j][] = PN; o_tst[j][] = PN; o_tend[j][] = PN
                j > 1 && (o_pcur[j-1][] = PN)
            else
                r = recent[end]
                pts = [proj(q, j) for q in r.pts if all(isfinite, q)]
                o_tlast[j][] = isempty(pts) ? PN : pts
                o_tst[j][]   = [proj((r.θ_old..., r.p0...), j)]
                o_tend[j][]  = isempty(pts) ? PN : [pts[end]]
                j > 1 && (o_pcur[j-1][] = [proj(state_after(r), j)])
            end
        end
        nr > 0 && (c_last[] = recent[end].acc ? :forestgreen : C_REJ)
    end

    "Oscillation periods (ω² = eigenvalues of M⁻¹Λ) and the stability limit of the leapfrog step."
    function draw_tlen!()
        _, C = exact_posterior(S[], prior())
        λ1, λ2 = eig_prod(inv2(massmatrix()), inv2(C))
        ω1, ω2 = sqrt(λ1), sqrt(λ2)
        ε = sl_eps.value[]; L = sl_L.value[]
        stable = ε * ω2 < 2
        # the leapfrog oscillates with ω̃ = (2/ε)·asin(εω/2) (slightly faster than ω)
        f = stable ? [L * 2asin(ε * ω / 2) / 2π for ω in (ω1, ω2)] : [NaN, NaN]
        res = stable && !tg_jit.active[] && any(x -> x > 0.3 && abs(x - round(2x) / 2) < 0.06, f)
        tlen.text[] = stable ?
            @sprintf("trajectory = %.2f, %.2f periods", f...) *
            (res ? "  –  RESONANCE (≈ multiple of ½): try jitter ε" : "  (¼ ideal, ½ reflection, 1 U-turn)") *
            @sprintf("  ·  stable for ε < %.3g", 2 / ω2) :
            @sprintf("UNSTABLE: ε > 2/ω_max = %.3g  –  all trajectories diverge", 2 / ω2)
        tlen.color = !stable ? C_REJ : res ? :darkorange3 : :gray35
    end

    function draw_chain!()
        P = prior(); Sd = S[]
        n  = nit(); kc, ac = ks[end], as[end]
        line(k, a) = [Point2f(XLIM[1], a + k * XLIM[1]), Point2f(XLIM[2], a + k * XLIM[2])]
        # data panel
        o_cur[] = line(kc, ac)
        i0 = max(1, length(ks) - NSPAG + 1)
        o_spag[] = reduce(vcat, [line(ks[i], as[i]) for i in i0:length(ks)])
        show1 = lastn[] == 1 && !isempty(recent)
        L = show1 ? recent[end] : nothing
        vis_prop[] = show1 && all(isfinite, L.θp)
        vis_prop[] && (o_prop[] = line(L.θp...))
        # map
        o_vis[]  = Point2f.(ks, as)
        rej = [p for (p, a) in zip(props, accs) if !a && all(isfinite, p)]
        o_rej[]  = isempty(rej) ? PN : [Point2f(p...) for p in rej]
        o_mcur[] = [Point2f(kc, ac)]
        draw_traj!()
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
        txt = @sprintf("iterations: %d   accepted: %d (%.0f %%)   divergent: %d   current (k, a) = (%s, %s)",
                       n, nacc, n == 0 ? 0.0 : 100nacc / n, ndiv[], fmt(kc), fmt(ac))
        if show1
            txt *= @sprintf("\nlast: p₀ = (%s, %s),  %d steps of ε = %s  →  θ′ = (%s, %s)", fmt2(L.p0[1]), fmt2(L.p0[2]),
                            sl_L.value[], fmt(sl_eps.value[]), fmt(L.θp[1]), fmt(L.θp[2]))
            txt *= "\n   H: " * fmtr(L.H0) * " → " * fmtr(L.H1) * "   ΔH = " * fmtr(L.ΔH) * "   α = " * fmtr(L.α) *
                   @sprintf(",  u = %.3g", L.u) * (L.div ? "  → DIVERGENT" : L.acc ? "  → ACCEPTED" : "  → REJECTED")
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
        axM.title = "misfit χ²(k, a) and trajectories  –  click to set the start"
        axK.title = isempty(kk) ? "marginal of k  (no samples yet)" : "marginal of k  ($(length(kk)) samples)"
        axA.title = "marginal of a"
        draw_tlen!()
    end

    function draw_limits!()
        k1, k2, a1, a2 = window()
        P = prior(); m, C = exact_posterior(S[], P); sk, sa = sd2(C)
        yk = max(maximum(p[2] for p in o_pk[] if k1 <= p[1] <= k2; init = 0.0), tg_exact.active[] ? gauss(0, 0, sk) : 0.0)
        ya = max(maximum(p[2] for p in o_pa[] if a1 <= p[1] <= a2; init = 0.0), tg_exact.active[] ? gauss(0, 0, sa) : 0.0)
        hk = vis_hist[] ? maximum(p[2] for p in o_hk[]) : 0.0
        ha = vis_hist[] ? maximum(p[2] for p in o_ha[]) : 0.0
        limits!(axK, k1, k2, 0, 1.12 * max(yk, min(hk, 2.5yk)) + 1e-9)
        limits!(axA, a1, a2, 0, 1.12 * max(ya, min(ha, 2.5ya)) + 1e-9)
        # phase planes: momentum range from p ~ N(0, M) and the recent trajectories
        M = massmatrix()
        pk = 4sqrt(M.xx); pa = 4sqrt(M.yy)
        for r in recent, q in r.pts
            all(isfinite, q) || continue
            k1 <= q[1] <= k2 && (pk = max(pk, abs(q[3])))
            a1 <= q[2] <= a2 && (pa = max(pa, abs(q[4])))
        end
        limits!(axPk, k1, k2, -1.1pk, 1.1pk)
        limits!(axPa, a1, a2, -1.1pa, 1.1pa)
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
        set_status!("Data initialised and χ²(k, a) computed. Choose the priors, ε, L and the mass matrix, then press \"Run HMC batch\".")
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
        t = findfirst(==(sel), MASSES)
        (t === nothing || t == mtype[]) && return
        mtype[] = t
        draw_chain!()
        if t > 1 && !learned[][2]
            set_status!("Mass matrix: M = I until you press \"learn Σ from chain\" (after a pilot run).", warn = true)
        else
            set_status!("Mass matrix: " * MASSES[t] * ". The target is unchanged, so the chain continues." *
                        (t == 3 ? " Try ε ≈ 0.3 and L ≈ 5 (a quarter period)." : ""))
        end
    end
    on(_ -> draw_tlen!(), sl_eps.value)
    on(_ -> draw_tlen!(), sl_L.value)
    on(_ -> draw_tlen!(), tg_jit.active)
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
                 learn = learn!, mass = t -> (menu.i_selected[] = t),
                 setstart = θ -> (start[] = θ; restart!("starting point changed")),
                 sliders = (kt = sl_kt, at = sl_at, sig = sl_sig, n = sl_n, mk = sl_mk, sk = sl_sk, ma = sl_ma, sa = sl_sa,
                            eps = sl_eps, L = sl_L, batch = sl_batch, burn = sl_burn),
                 toggles = (zoom = tg_zoom, exact = tg_exact, prior = tg_prior, rej = tg_rej, old = tg_old, vis = tg_vis, jitter = tg_jit),
                 ks = ks, as = as, ndiv = ndiv)
end

# =============================================================================
# RUN
# =============================================================================
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    fig, app = build_app()
    screen = display(fig)
    isinteractive() || wait(screen)     # keep the window open when run from a terminal
end
