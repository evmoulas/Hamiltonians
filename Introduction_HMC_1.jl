#=
Introduction_HMC_1 – Hamiltonian Monte Carlo for the fit of y = k·x
==================================================================

Same problem as Introduction_MCMC_1:  data yᵢ = k_true·xᵢ + σ·εᵢ ,  model y = k·x,
  χ²(k) = Σᵢ (yᵢ − k·xᵢ)² / σ² ,   prior k ~ N(μₚ, σₚ²),
  posterior  p(k | d) ∝ exp(−χ²(k)/2) · exp(−(k − μₚ)²/(2σₚ²)).

Hamiltonian Monte Carlo (HMC)
  The parameter k is treated as the POSITION of a particle that moves in the
  potential (the "bowl")
        U(k) = −ln p(k|d) = ½ χ²(k) + ½ (k − μₚ)²/σₚ² + const ,
  and an auxiliary MOMENTUM p is added, with kinetic energy p²/(2M) (M = mass).
  The total energy is the Hamiltonian
        H(k, p) = U(k) + p²/(2M) ,
  and exp(−H) = p(k|d) · N(p; 0, M): k and p are independent, k has the posterior.

  One HMC iteration:
    1. draw a new momentum  p₀ ~ N(0, M)         (vertical jump in phase space)
    2. follow Hamilton's equations  dk/dt = p/M ,  dp/dt = −dU/dk  for a time L·ε
       with L LEAPFROG steps of size ε:
           p ← p − ½ε·U′(k);   k ← k + ε·p/M;   p ← p − ½ε·U′(k)
       (the particle rolls in the bowl along a contour of H)
    3. accept the end point (k′, p′) with probability
           α = min(1, exp(−[H(k′, p′) − H(k, p₀)])) = min(1, e^(−ΔH))
  The exact dynamics conserve H (ΔH = 0, α = 1); the leapfrog scheme conserves
  it only approximately, ΔH grows with ε. Because leapfrog is reversible and
  preserves phase-space volume, no Hastings correction is needed.
  Here U′(k) = −Σ xᵢ(yᵢ − k·xᵢ)/σ² + (k − μₚ)/σₚ²  (HMC needs the gradient).

Why HMC: a trajectory can travel far (to the other side of the bowl) and still
be accepted with high probability, so successive samples are much less
correlated than for a random walk.

Panels
  upper left       : data, true line, current fit, last states of the chain
  upper right (1)  : misfit χ²(k), visited states and the last trajectory on it
  upper right (2)  : PHASE SPACE (k, p): contours of H, the recent trajectories
                     (Hamiltonian flow along the contours) and the momentum
                     refreshes (vertical dotted jumps at fixed k)
  lower left       : controls
  lower right      : prior / posterior histogram and trace plot

What to try
  * ε small, L·ε about a QUARTER of an oscillation period (e.g. ε = 0.02, L = 8):
    almost every trajectory is accepted and gives a nearly independent sample.
  * L·ε ≈ half a period: the particle ends on the opposite side, k → 2m − k
    (strongly anti-correlated trace); the distance from the mean hardly changes,
    so from a start far away the chain only swings back and forth.
  * ε too large: ΔH grows, many rejections. Above the stability limit of the
    leapfrog scheme, ε > 2/ω = 2·σ_post·√M, every trajectory diverges.
  * the mass M rescales time (ω = 1/(σ_post·√M)): a larger M allows a larger ε.
  * L·ε = one full period: the particle comes back to where it started (U-turn),
    the chain hardly moves – the reason for NUTS ("No-U-Turn Sampler").
  * Start far away (k₀ = −0.5): the energy is huge, the first trajectories swing
    across the whole bowl; energy is lost only through momentum refreshes.

Run from a terminal:      julia Introduction_HMC_1.jl
or from the Julia REPL:   include("Introduction_HMC_1.jl")
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
const KLIM     = (-1.0, 4.0)        # k range of the misfit plot (full view)
const NK       = 1001               # grid points for χ²(k)
const NMAX     = 50                 # largest number of data points
const SEED     = 1                  # seed of the first noise realisation

const K_TRUE   = 1.5                # default true slope
const SIGMA    = 2.0                # default noise standard deviation
const NDATA    = 12                 # default number of data points
const K_START  = -0.5               # default starting guess
const PRIOR_MU = 0.5                # default prior mean
const PRIOR_SD = 1.0                # default prior standard deviation
const EPS      = 0.02               # default leapfrog step ε
const NLEAP    = 20                 # default number of leapfrog steps L
const MASS     = 1.0                # default mass M
const BATCHES  = [1, 10, 100, 1000] # batch sizes of the "Run" button
const NSPAG    = 40                 # number of last chain states drawn in the data panel
const NTRAJ    = 25                 # number of last trajectories drawn in phase space
const NPH      = 161                # grid (NPH × NPH) for the contours of H
const DIVERGE  = 1000.0             # ΔH above this: divergent trajectory

const C_DATA  = :black
const C_TRUE  = :gray45
const C_CUR   = :darkorange2        # current state of the chain
const C_CHAIN = :seagreen           # visited states / histogram / accepted trajectories
const C_REJ   = :red3               # rejected
const C_PRIOR = :royalblue3
const C_POST  = :black              # exact posterior
const C_MIS   = :purple4            # misfit
const C_KICK  = :gray30             # momentum refresh

# =============================================================================
# MATHEMATICS (independent of the plotting)
# =============================================================================
"Sums of the data needed for χ²: Σx², Σxy, Σy² and σ²."
struct DataSums
    xx::Float64
    xy::Float64
    yy::Float64
    s2::Float64
end
DataSums(x, y, σ) = DataSums(sum(abs2, x), sum(x .* y), sum(abs2, y), σ^2)

"Misfit χ²(k) = Σ (yᵢ − k xᵢ)² / σ²  (expanded, so it costs O(1))."
chi2(k, S::DataSums) = max(S.yy - 2k * S.xy + k^2 * S.xx, 0.0) / S.s2

gauss(k, m, s) = exp(-0.5 * ((k - m) / s)^2) / (s * sqrt(2π))

"Exact Gaussian posterior (mean, standard deviation) of the linear model."
function exact_posterior(S::DataSums, μ, s)
    A = S.xx / S.s2 + 1 / s^2
    B = S.xy / S.s2 + μ / s^2
    return B / A, 1 / sqrt(A)
end

"Least-squares slope k̂ (minimum of χ²) and its standard error."
lsq(S::DataSums) = (S.xy / S.xx, sqrt(S.s2 / S.xx))

"Target and HMC settings: data, prior, leapfrog step ε, number of steps L, mass M."
struct HMC
    S::DataSums
    μ::Float64
    sp::Float64
    ε::Float64
    L::Int
    M::Float64
end

"Potential energy U(k) = −ln p(k|d) = ½χ² + ½(k−μₚ)²/σₚ²."
U(k, P::HMC) = 0.5 * chi2(k, P.S) + 0.5 * ((k - P.μ) / P.sp)^2
"Its derivative U′(k) = −Σx(y − kx)/σ² + (k − μₚ)/σₚ²."
dU(k, P::HMC) = -(P.S.xy - k * P.S.xx) / P.S.s2 + (k - P.μ) / P.sp^2
"Hamiltonian H(k, p) = U(k) + p²/(2M)."
H(k, p, P::HMC) = U(k, P) + p^2 / (2P.M)

"""
L leapfrog steps from (k, p). Returns the phase-space points after every step
(momentum synchronised with the position) and the end point.
"""
function leapfrog(k, p, P::HMC)
    pts = [Point2f(k, p)]
    ph  = p - 0.5P.ε * dU(k, P)                    # half step in p
    g   = 0.0
    for i in 1:P.L
        k += P.ε * ph / P.M                        # full step in k
        g  = dU(k, P)
        push!(pts, Point2f(k, ph - 0.5P.ε * g))    # synchronised p (for plotting)
        (!isfinite(k) || abs(k) > 1e6) && return pts, k, NaN
        i < P.L && (ph -= P.ε * g)                 # full step in p
    end
    return pts, k, ph - 0.5P.ε * g                 # last half step in p
end

"One HMC iteration from k."
function hmc_step(rng, k, P::HMC)
    p0 = sqrt(P.M) * randn(rng)
    pts, k1, p1 = leapfrog(k, p0, P)
    H0 = H(k, p0, P)
    H1 = isfinite(k1) && isfinite(p1) ? H(k1, p1, P) : Inf
    ΔH = H1 - H0
    div = !(ΔH < DIVERGE)
    α  = div ? 0.0 : exp(min(0.0, -ΔH))
    u  = rand(rng)
    return (k_old = k, p0 = p0, kp = k1, pp = p1, pts = pts, H0 = H0, H1 = H1, ΔH = ΔH, α = α, u = u,
            acc = !div && u < α, div = div)
end

"Density histogram of v with about √n bins (at most 60). Returns (centres, densities, width)."
function density_hist(v)
    n = length(v)
    n == 0 && return [0.0], [0.0], 1.0
    lo, hi = extrema(v)
    if hi - lo < 1e-12
        w = 0.01 * max(abs(lo), 1.0)
        return [lo], [1 / w], w
    end
    nb = clamp(round(Int, sqrt(n)), 5, 60)
    w  = (hi - lo) / nb
    c  = zeros(nb)
    @inbounds for x in v
        c[clamp(floor(Int, (x - lo) / w) + 1, 1, nb)] += 1
    end
    return collect(lo .+ w .* ((1:nb) .- 0.5)), c ./ (n * w), w
end

fmt(v) = @sprintf("%.4g", abs(v) < 1e-9 ? 0.0 : v)
fmtr(v) = (v == 0 || 1e-3 <= abs(v) < 1e4) ? @sprintf("%.3g", v) : @sprintf("%.2e", v)

# =============================================================================
# APPLICATION
# =============================================================================
function build_app()
    # ---------------------------------------------------------------- state
    kgrid  = collect(range(KLIM..., length = NK))
    rng_d  = Ref(Xoshiro(SEED))                       # data (x positions and noise)
    rng_c  = Xoshiro(SEED + 1000)                     # Markov chain
    xall   = Ref(zeros(NMAX)); eall = Ref(zeros(NMAX))
    xd     = Ref(Float64[]);   yd   = Ref(Float64[])
    S      = Ref(DataSums(1.0, 0.0, 0.0, 1.0))
    used   = Ref((K_TRUE, SIGMA, NDATA))              # data settings in use
    chain  = Float64[]                                # visited states (chain[1] = k₀)
    props  = Float64[]                                # end points of all trajectories
    accs   = Bool[]                                   # accepted?
    ndiv   = Ref(0)                                   # number of divergent trajectories
    recent = Any[]                                    # the last NTRAJ iterations (for phase space)
    lastn  = Ref(0)                                   # size of the last batch

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Introduction to HMC 1  –  Hamiltonian Monte Carlo  (Bayesian fit of  y = k·x)",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(540))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")        # no fixed height: it takes the space the controls leave
    top = right[1, 1] = GridLayout()
    axM = Axis(top[1, 1]; xlabel = "k", ylabel = "χ²(k)",
               ytickformat = vs -> [@sprintf("%.3g", v) for v in vs])
    axH = Axis(top[1, 2]; xlabel = "k   (position)", ylabel = "p   (momentum)")
    colgap!(top, 14)
    axP = Axis(right[2, 1]; xlabel = "k", ylabel = "probability density")
    axT = Axis(right[3, 1]; xlabel = "iteration", ylabel = "k", title = "trace plot")
    rowsize!(right, 1, Auto(1.5))
    rowgap!(right, 8)
    for a in (axD, axM, axH, axP, axT), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
    end

    # display options (below the trace plot, to keep the left column short)
    tg = right[4, 1] = GridLayout(halign = :center, tellwidth = false)
    tg_zoom  = Toggle(tg[1, 1], active = false); Label(tg[1, 2], "zoom on posterior", halign = :left)
    tg_log   = Toggle(tg[1, 3], active = false); Label(tg[1, 4], "log scale χ²", halign = :left)
    tg_exact = Toggle(tg[1, 5], active = true);  Label(tg[1, 6], "exact posterior", halign = :left)
    tg_rej   = Toggle(tg[2, 1], active = true);  Label(tg[2, 2], "rejected trajectories", halign = :left)
    tg_cont  = Toggle(tg[2, 3], active = true);  Label(tg[2, 4], "contours of H", halign = :left)
    colgap!(tg, 6); rowgap!(tg, 2)
    for c in (2, 4)
        colgap!(tg, c, 18)
    end

    # --- 1. data
    Label(left[2, 1], "1 · synthetic data   y = k_true·x + σ·ε", font = :bold, halign = :left, tellwidth = false)
    sgD = SliderGrid(left[3, 1],
        (label = "k_true", range = 0:0.05:3, startvalue = K_TRUE, format = "{:.2f}"),
        (label = "noise σ", range = 0.1:0.1:5, startvalue = SIGMA, format = v -> @sprintf("%.1f  (σ² = %.3g)", v, v^2)),
        (label = "N points", range = 3:NMAX, startvalue = NDATA, format = "{:d}");
        tellheight = true, width = 500, halign = :left)
    sl_kt, sl_sig, sl_n = sgD.sliders
    bD = left[4, 1] = GridLayout(halign = :left)
    btn_init  = Button(bD[1, 1], label = "Initialise data (compute χ²)", buttoncolor = RGBf(0.80, 0.88, 1.0))
    btn_noise = Button(bD[1, 2], label = "new noise realisation")

    # --- 2. prior and start
    Label(left[5, 1], "2 · starting guess and prior   k ~ N(μₚ, σₚ²)", font = :bold, halign = :left, tellwidth = false)
    nice   = sort(unique(round.(vcat([m .* 10.0^e for e in -3:3 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3)))
    sgP = SliderGrid(left[6, 1],
        (label = "start k₀", range = KLIM[1]:0.05:KLIM[2], startvalue = K_START, format = "{:.2f}"),
        (label = "prior mean μₚ", range = KLIM[1]:0.05:KLIM[2], startvalue = PRIOR_MU, format = "{:.2f}"),
        (label = "prior st.dev. σₚ", range = filter(v -> 0.02 <= v <= 20, nice), startvalue = PRIOR_SD,
         format = v -> @sprintf("%.3g  (σₚ² = %.3g)", v, v^2));
        tellheight = true, width = 500, halign = :left)
    sl_k0, sl_mu, sl_sp = sgP.sliders

    # --- 3. sampler
    Label(left[7, 1], "3 · Hamiltonian Monte Carlo   H(k, p) = U(k) + p²/(2M)", font = :bold,
          halign = :left, tellwidth = false)
    sgS = SliderGrid(left[8, 1],
        (label = "leapfrog step ε", range = filter(v -> 0.001 <= v <= 1, nice), startvalue = EPS, format = "{:.3g}"),
        (label = "leapfrog steps L", range = 1:100, startvalue = NLEAP, format = "{:d}"),
        (label = "mass M", range = filter(v -> 0.01 <= v <= 1000, nice), startvalue = MASS, format = "{:.3g}"),
        (label = "batch size", range = BATCHES, startvalue = BATCHES[1], format = "{:d} samples"),
        (label = "burn-in", range = 0:10:2000, startvalue = 0, format = "discard first {:d}");
        tellheight = true, width = 500, halign = :left)
    sl_eps, sl_L, sl_M, sl_batch, sl_burn = sgS.sliders
    bS = left[9, 1] = GridLayout(halign = :left)
    btn_run   = Button(bS[1, 1], label = "Run HMC batch", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 220)
    btn_reset = Button(bS[1, 2], label = "Reset chain")
    tlen      = Label(bS[1, 3], " ", halign = :left, color = :gray35, fontsize = 11)

    # fixed heights (6 and 2 lines), so that the data axis does not jump when the text changes
    info   = Label(left[10, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   fontsize = 13, height = 96)
    status = Label(left[11, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 34)
    rowgap!(left, 4)
    for sg in (sgD, sgP, sgS)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    P0 = Point2f[(0, 0)]
    PN = Point2f[(NaN, NaN)]
    # data panel
    o_dat  = Observable(P0)
    o_err  = Observable(Point2f[(0, 0), (0, 0)])
    o_true = Observable(Point2f[(0, 0), (1, 0)])
    o_spag = Observable(Point2f[(0, 0), (0, 0)])
    o_cur  = Observable(Point2f[(0, 0), (1, 0)])
    o_prop = Observable(Point2f[(0, 0), (1, 0)]); c_last = Observable{Any}(C_CHAIN); vis_prop = Observable(false)
    # misfit panel
    o_mis  = Observable(P0); o_mpo = Observable(P0)
    o_vis  = Observable(P0); o_rej = Observable(P0)
    o_mcur = Observable(P0); o_mtraj = Observable(PN)
    o_kt   = Observable(K_TRUE); o_khat = Observable(K_TRUE)
    # phase space
    o_hx = Observable(collect(range(0, 1, length = NPH))); o_hy = Observable(collect(range(0, 1, length = NPH)))
    o_hz = Observable(zeros(NPH, NPH)); o_lev = Observable(collect(1.0:10))
    o_tacc = Observable(PN); o_trej = Observable(PN)            # older trajectories (NaN separated)
    o_kick = Observable(PN)                                     # momentum refreshes (segments)
    o_tlast = Observable(PN)                                    # last trajectory
    o_tstart = Observable(PN); o_tend = Observable(PN)
    o_hcur = Observable(PN)
    # posterior panel
    o_prior = Observable(P0); o_exact = Observable(P0)
    o_hist  = Observable(P0); o_bw = Observable(1.0); vis_hist = Observable(false)
    # trace panel
    o_trace = Observable(P0); o_burn = Observable(Rect2f(0, 0, 0, 1)); o_pm = Observable(0.0)
    o_band  = Observable(Rect2f(0, 0, 1, 1))

    # upper left: data
    lines!(axD, o_true; color = C_TRUE, linewidth = 2, linestyle = :dash, label = "true line  y = k_true·x")
    linesegments!(axD, o_spag; color = (C_CHAIN, 0.25), linewidth = 1, label = "last $NSPAG states of the chain")
    lines!(axD, o_prop; color = c_last, linewidth = 2, linestyle = :dot, visible = vis_prop, label = "proposal k′")
    lines!(axD, o_cur; color = C_CUR, linewidth = 3, label = "current fit  y = k·x")
    linesegments!(axD, o_err; color = (C_DATA, 0.6), linewidth = 1.2)
    scatter!(axD, o_dat; color = C_DATA, markersize = 9, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 11, rowgap = 0, colgap = 6, nbanks = 2, patchsize = (16, 10),
               padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

    # upper right (1): misfit
    vlines!(axM, o_kt; color = C_TRUE, linestyle = :dash, linewidth = 1.5, label = "k_true")
    vlines!(axM, o_khat; color = C_MIS, linestyle = :dot, linewidth = 1.5, label = "least squares k̂")
    lines!(axM, o_mpo; color = (C_PRIOR, 0.8), linewidth = 2, linestyle = :dash, label = "χ² + (k−μₚ)²/σₚ² = 2U(k)")
    lines!(axM, o_mis; color = C_MIS, linewidth = 3, label = "misfit χ²(k)")
    scatter!(axM, o_rej; color = (C_REJ, 0.35), marker = :xcross, markersize = 7, visible = tg_rej.active,
             label = "rejected end points")
    scatter!(axM, o_vis; color = (C_CHAIN, 0.45), markersize = 7, label = "visited states")
    scatterlines!(axM, o_mtraj; color = c_last, markersize = 5, linewidth = 1.2, label = "last trajectory")
    scatter!(axM, o_mcur; color = C_CUR, markersize = 16, strokecolor = :black, strokewidth = 1.5,
             label = "current state")
    axislegend(axM; position = :ct, labelsize = 10, rowgap = 0, padding = (5, 5, 3, 3), patchsize = (14, 8),
               backgroundcolor = (:white, 0.85))

    # upper right (2): phase space
    contour!(axH, o_hx, o_hy, o_hz; levels = o_lev, color = (:gray55, 0.7), linewidth = 0.8, visible = tg_cont.active)
    lines!(axH, o_trej; color = (C_REJ, 0.35), linewidth = 1, linestyle = :dash, visible = tg_rej.active)
    lines!(axH, o_tacc; color = (C_CHAIN, 0.45), linewidth = 1.2)
    linesegments!(axH, o_kick; color = (C_KICK, 0.6), linewidth = 1, linestyle = :dot)
    scatterlines!(axH, o_tlast; color = c_last, markersize = 5, linewidth = 2.2)
    scatter!(axH, o_tstart; color = :white, strokecolor = c_last, strokewidth = 2, markersize = 11)
    scatter!(axH, o_tend; color = c_last, marker = :diamond, markersize = 13, strokecolor = :black, strokewidth = 1)
    scatter!(axH, o_hcur; color = C_CUR, markersize = 16, strokecolor = :black, strokewidth = 1.5)

    # lower right: prior / posterior
    barplot!(axP, o_hist; width = o_bw, gap = 0, color = (C_CHAIN, 0.45), strokecolor = (C_CHAIN, 0.9),
             strokewidth = 0.5, visible = vis_hist, label = "histogram of the samples (after burn-in)")
    lines!(axP, o_prior; color = C_PRIOR, linewidth = 2.5, label = "prior  N(μₚ, σₚ²)")
    lines!(axP, o_exact; color = C_POST, linewidth = 2, linestyle = :dash, visible = tg_exact.active,
           label = "exact posterior")
    vlines!(axP, o_kt; color = C_TRUE, linestyle = :dash, linewidth = 1.5)
    axislegend(axP; position = :rt, labelsize = 11, rowgap = 0, padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

    # lower right: trace
    poly!(axT, o_band; color = (C_POST, 0.08), visible = tg_exact.active)
    hlines!(axT, o_pm; color = C_POST, linestyle = :dash, linewidth = 1.2, visible = tg_exact.active)
    poly!(axT, o_burn; color = (:gray50, 0.18))
    hlines!(axT, o_kt; color = C_TRUE, linestyle = :dash, linewidth = 1.5)
    lines!(axT, o_trace; color = C_CHAIN, linewidth = 1.2)

    # ------------------------------------------------------------ functions
    prior_pars() = (sl_mu.value[], sl_sp.value[])
    hmc() = HMC(S[], prior_pars()..., sl_eps.value[], sl_L.value[], sl_M.value[])

    "Draw new x positions and new standard-normal noise (the noise is scaled by σ later)."
    function new_noise!()
        rng_d[] = Xoshiro(rand(UInt32))
        xall[]  = XLIM[1] .+ (XLIM[2] - XLIM[1]) .* rand(rng_d[], NMAX)
        eall[]  = randn(rng_d[], NMAX)
    end

    "Data from the sliders and the stored noise, then the misfit curve χ²(k) on the k grid."
    function init_data!()
        kt, σ, n = sl_kt.value[], sl_sig.value[], sl_n.value[]
        used[] = (kt, σ, n)
        xd[] = xall[][1:n]
        yd[] = kt .* xd[] .+ σ .* eall[][1:n]
        S[]  = DataSums(xd[], yd[], σ)
        o_kt[] = kt
        o_khat[] = lsq(S[])[1]
        o_dat[]  = Point2f.(xd[], yd[])
        o_err[]  = vec(permutedims(hcat(Point2f.(xd[], yd[] .- σ), Point2f.(xd[], yd[] .+ σ))))
        o_true[] = [Point2f(XLIM[1], kt * XLIM[1]), Point2f(XLIM[2], kt * XLIM[2])]
        o_mis[]  = Point2f.(kgrid, chi2.(kgrid, Ref(S[])))
        axD.title = @sprintf("data: N = %d,  k_true = %.2f,  σ = %.2g", n, kt, σ)
        lo, hi = extrema(vcat(yd[] .- σ, yd[] .+ σ, 0.0))
        limits!(axD, XLIM[1] - 0.3, XLIM[2] + 0.3, lo - 0.08(hi - lo), hi + 0.08(hi - lo))
    end

    "Restart the chain at k₀."
    function reset_chain!()
        empty!(chain); empty!(props); empty!(accs); empty!(recent)
        push!(chain, sl_k0.value[])
        ndiv[] = 0; lastn[] = 0
    end

    "Run n HMC iterations."
    function run_batch!(n)
        P = hmc()
        k = chain[end]
        for _ in 1:n
            r = hmc_step(rng_c, k, P)
            push!(props, r.kp); push!(accs, r.acc)
            r.div && (ndiv[] += 1)
            r.acc && (k = r.kp)
            push!(chain, k)
            push!(recent, r)
            length(recent) > NTRAJ && popfirst!(recent)
        end
        lastn[] = n
    end

    "k window of the k axes (full range or zoomed on the posterior)."
    function kwindow()
        m, sd = exact_posterior(S[], prior_pars()...)
        tg_zoom.active[] ? (m - 6sd, m + 6sd) : KLIM
    end

    "Everything that depends on the prior (curves only)."
    function draw_prior!()
        μ, s = prior_pars()
        m, sd = exact_posterior(S[], μ, s)
        o_mpo[]   = Point2f.(kgrid, chi2.(kgrid, Ref(S[])) .+ ((kgrid .- μ) ./ s) .^ 2)
        k1, k2 = kwindow()
        kk = sort(vcat(range(min(k1, KLIM[1]), max(k2, KLIM[2]), length = 4001), range(m - 6sd, m + 6sd, length = 801)))
        o_prior[] = Point2f.(kk, gauss.(kk, μ, s))
        o_exact[] = Point2f.(kk, gauss.(kk, m, sd))
        o_pm[]    = m
    end

    "State of the chain in phase space after iteration r: end point if accepted, start point otherwise."
    state_after(r) = r.acc ? Point2f(r.kp, r.pp) : Point2f(r.k_old, r.p0)

    "Phase-space drawing: recent trajectories, momentum refreshes, last trajectory."
    function draw_phase!()
        acc = Point2f[]; rej = Point2f[]; kick = Point2f[]
        nr = length(recent)
        for (i, r) in enumerate(recent)
            if i < nr                                              # older trajectories
                pts = filter(p -> all(isfinite, p), r.pts)
                append!(r.acc ? acc : rej, pts); push!(r.acc ? acc : rej, Point2f(NaN, NaN))
            end
            if i > 1                                               # refresh: same k, new momentum
                s = state_after(recent[i-1])
                push!(kick, s, Point2f(s[1], r.p0))
            end
        end
        o_tacc[] = isempty(acc) ? PN : acc
        o_trej[] = isempty(rej) ? PN : rej
        o_kick[] = isempty(kick) ? Point2f[(NaN, NaN), (NaN, NaN)] : kick
        if nr == 0
            o_tlast[] = PN; o_tstart[] = PN; o_tend[] = PN; o_mtraj[] = PN
            o_hcur[]  = [Point2f(chain[end], 0)]
        else
            r = recent[end]
            pts = filter(p -> all(isfinite, p), r.pts)
            c_last[]  = r.acc ? :forestgreen : C_REJ
            o_tlast[] = isempty(pts) ? PN : pts
            o_tstart[] = [Point2f(r.k_old, r.p0)]
            o_tend[]   = [pts[end]]
            o_mtraj[]  = [Point2f(p[1], chi2(p[1], S[])) for p in pts]
            o_hcur[]   = [state_after(r)]
        end
    end

    "Everything that depends on the chain."
    function draw_chain!()
        μ, s = prior_pars()
        Sd = S[]
        n  = length(chain) - 1
        kc = chain[end]
        # data panel
        o_cur[] = [Point2f(XLIM[1], kc * XLIM[1]), Point2f(XLIM[2], kc * XLIM[2])]
        sp = chain[max(1, end - NSPAG + 1):end]
        o_spag[] = vec(permutedims(hcat(Point2f.(XLIM[1], sp .* XLIM[1]), Point2f.(XLIM[2], sp .* XLIM[2]))))
        show1 = lastn[] == 1 && !isempty(recent)
        L = show1 ? recent[end] : nothing
        vis_prop[] = show1 && isfinite(L.kp)
        show1 && isfinite(L.kp) && (o_prop[] = [Point2f(XLIM[1], L.kp * XLIM[1]), Point2f(XLIM[2], L.kp * XLIM[2])])
        # misfit panel
        o_vis[]  = Point2f.(chain, chi2.(chain, Ref(Sd)))
        rej = filter(isfinite, props[.!accs])
        o_rej[]  = isempty(rej) ? PN : Point2f.(rej, chi2.(rej, Ref(Sd)))
        o_mcur[] = [Point2f(kc, chi2(kc, Sd))]
        draw_phase!()
        # posterior panel: histogram after burn-in
        nb   = sl_burn.value[]
        kept = n > nb ? chain[nb+2:end] : Float64[]          # chain[1] is k₀ (iteration 0)
        c, d, w = density_hist(kept)
        o_hist[] = Point2f.(c, d); o_bw[] = w
        vis_hist[] = !isempty(kept)
        # trace panel
        o_trace[] = Point2f.(0:n, chain)
        draw_limits!()
        # text
        m, sd = exact_posterior(Sd, μ, s)
        kh, kse = lsq(Sd)
        nacc = count(accs)
        txt = @sprintf("iterations: %d    accepted: %d (%.0f %%)    divergent: %d    current k = %s",
                       n, nacc, n == 0 ? 0.0 : 100nacc / n, ndiv[], fmt(kc))
        if show1
            txt *= @sprintf("\nlast step: p₀ = %s,  %d leapfrog steps of ε = %s  →  k′ = %s,  p′ = %s",
                            fmt(L.p0), sl_L.value[], fmt(sl_eps.value[]), fmt(L.kp), fmt(L.pp))
            txt *= "\n   H = U + p²/2M:  " * fmtr(L.H0) * " → " * fmtr(L.H1) * "    ΔH = " * fmtr(L.ΔH)
            txt *= @sprintf("\n   α = min(1, e^(−ΔH)) = %s,   u = %.3g   →  %s", fmtr(L.α), L.u,
                            L.div ? "DIVERGENT trajectory: rejected" :
                            L.acc ? "u < α: ACCEPTED (move to k′)" : "u ≥ α: REJECTED (stay at k, k repeated)")
        else
            txt *= "\n\n\n"
        end
        txt *= isempty(kept) ? "\nsamples after burn-in: none yet" :
               @sprintf("\nsamples after burn-in: %d     mean = %s,  st.dev. = %s", length(kept),
                        fmt(sum(kept) / length(kept)),
                        fmt(length(kept) > 1 ? sqrt(sum(abs2, kept .- sum(kept) / length(kept)) / (length(kept) - 1)) : 0.0))
        txt *= @sprintf("\nexact posterior: %s ± %s     least squares k̂ = %s ± %s", fmt(m), fmt(sd), fmt(kh), fmt(kse))
        info.text[] = txt
        axP.title = "prior and posterior  (" * (isempty(kept) ? "no samples yet" : "$(length(kept)) samples") * ")"
        axM.title = "misfit χ²(k)  –  click to set k₀"
        axH.title = "phase space:  contours of H,  last $(min(length(recent), NTRAJ)) trajectories"
    end

    "Trajectory length and the oscillation period of the particle at the bottom of the bowl."
    function draw_tlen!()
        μ, s = prior_pars()
        m, sd = exact_posterior(S[], μ, s)
        ω = 1 / (sd * sqrt(sl_M.value[]))             # U ≈ (k−m)²/(2sd²): ω = 1/(sd·√M)
        ε = sl_eps.value[]; L = sl_L.value[]
        stable = ε * ω < 2
        f = stable ? L * 2asin(ε * ω / 2) / 2π : NaN    # leapfrog frequency ω̃ = (2/ε)·asin(εω/2)
        tlen.text[] = (stable ? @sprintf("trajectory = %.2f periods\n(¼ ideal, ½ reflection, 1 U-turn)", f) :
                                "UNSTABLE ε") * @sprintf("\nstable for ε < 2/ω = %.3g", 2 / ω)
        tlen.color = !stable ? C_REJ : (f > 0.3 && abs(f - round(2f) / 2) < 0.06) ? :darkorange3 : :gray35
    end

    "Axis limits (k window shared by the k axes) and the contours of H."
    function draw_limits!()
        k1, k2 = kwindow()
        Sd = S[]; μ, s = prior_pars()
        # misfit
        ks = range(k1, k2, length = 401)
        cv = chi2.(ks, Ref(Sd))
        cmin = minimum(cv); cmax = maximum(cv)
        if tg_log.active[]
            limits!(axM, k1, k2, max(cmin, 1e-3) / 2, cmax * 3)   # positive limits first, then the scale
            axM.yscale = log10
        else
            axM.yscale = identity
            limits!(axM, k1, k2, 0, cmax * 1.12 + 1e-9)
        end
        # phase space: p range from the momentum distribution and the recent trajectories
        P  = hmc()
        pm = 4 * sqrt(P.M)
        for r in recent, p in r.pts
            isfinite(p[1]) && isfinite(p[2]) && k1 <= p[1] <= k2 && (pm = max(pm, abs(p[2])))
        end
        pm *= 1.1
        limits!(axH, k1, k2, -pm, pm)
        hx = collect(range(k1, k2, length = NPH)); hy = collect(range(-pm, pm, length = NPH))
        hz = [H(k, p, P) for k in hx, p in hy]
        m, sd = exact_posterior(Sd, μ, s)
        Hmin = U(m, P)
        o_hx[] = hx; o_hy[] = hy                       # the grid size never changes: no size mismatch
        o_lev[] = Hmin .+ 0.5 .* 2.0 .^ (0:16)         # H − H_min = 0.5, 1, 2, 4, … (dense near the bottom)
        o_hz[] = hz
        # posterior (very tall histograms are clipped)
        ks2 = range(k1, k2, length = 2001)
        ycurve = max(maximum(gauss.(ks2, μ, s)), tg_exact.active[] ? gauss(m, m, sd) : 0.0)
        yhist  = vis_hist[] ? maximum(p[2] for p in o_hist[]) : 0.0
        limits!(axP, k1, k2, 0, 1.15 * max(ycurve, min(yhist, 2.5ycurve)) + 1e-9)
        # trace
        n = length(chain) - 1
        limits!(axT, -0.02 * max(n, 10), max(n, 10) * 1.02, k1, k2)
        o_burn[] = Rect2f(0, k1, min(sl_burn.value[], n), k2 - k1)
        o_band[] = Rect2f(0, m - sd, max(n, 10) * 1.02, 2sd)
        draw_tlen!()
    end

    function data_pending()
        (sl_kt.value[], sl_sig.value[], sl_n.value[]) != used[]
    end
    function set_status!(msg; warn = false)
        status.text[] = msg
        status.color = warn ? :red3 : :gray20
    end

    function initialise!()
        init_data!(); reset_chain!(); draw_prior!(); draw_chain!()
        set_status!("Data initialised and χ²(k) computed. Choose k₀, the prior, ε, L and M, then press \"Run HMC batch\".")
    end
    function restart!(why)
        reset_chain!(); draw_prior!(); draw_chain!()
        set_status!(why * " → chain restarted at k₀ = " * fmt(chain[1]) * ".")
    end

    # ------------------------------------------------------------ callbacks
    on(_ -> initialise!(), btn_init.clicks)
    on(btn_noise.clicks) do _
        new_noise!(); initialise!()
        set_status!("New noise realisation (new x positions and ε). Data and χ²(k) recomputed.")
    end
    for sl in (sl_kt, sl_sig, sl_n)
        on(sl.value) do _
            data_pending() ? set_status!("Data settings changed – press \"Initialise data\" to apply them.", warn = true) :
                             set_status!(" ")
        end
    end
    on(_ -> restart!("starting guess changed"), sl_k0.value)
    on(_ -> restart!("prior changed (new target posterior)"), sl_mu.value)
    on(_ -> restart!("prior changed (new target posterior)"), sl_sp.value)
    on(_ -> restart!("chain reset"), btn_reset.clicks)
    on(btn_run.clicks) do _
        run_batch!(sl_batch.value[]); draw_chain!()
        data_pending() && set_status!("Note: the data sliders were changed but not applied (press \"Initialise data\").",
                                      warn = true)
    end
    on(_ -> draw_tlen!(), sl_eps.value)
    on(_ -> draw_tlen!(), sl_L.value)
    on(_ -> draw_limits!(), sl_M.value)                # mass changes the contours of H
    on(_ -> draw_chain!(), sl_burn.value)
    on(_ -> (draw_prior!(); draw_chain!()), tg_zoom.active)
    on(_ -> draw_limits!(), tg_log.active)
    on(_ -> draw_limits!(), tg_exact.active)
    register_interaction!(axM, :pick_k0) do event::MouseEvent, _
        if event.type === MouseEventTypes.leftclick
            set_close_to!(sl_k0, clamp(event.data[1], KLIM...))
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
                 sliders = (kt = sl_kt, sig = sl_sig, n = sl_n, k0 = sl_k0, mu = sl_mu, sp = sl_sp,
                            eps = sl_eps, L = sl_L, M = sl_M, batch = sl_batch, burn = sl_burn),
                 toggles = (zoom = tg_zoom, log = tg_log, exact = tg_exact, rej = tg_rej, cont = tg_cont),
                 chain = chain)
end

# =============================================================================
# RUN
# =============================================================================
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    fig, app = build_app()
    screen = display(fig)
    isinteractive() || wait(screen)     # keep the window open when run from a terminal
end
