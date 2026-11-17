#=
Introduction_MCMC_1 – Bayesian fitting of a line through the origin with MCMC
============================================================================

Model and data
  We fit  y = k·x  (one unknown parameter k) to N data points (xᵢ, yᵢ) that were
  produced as   yᵢ = k_true·xᵢ + σ·εᵢ ,   εᵢ ~ N(0, 1)   (σ² = noise variance).

Misfit, likelihood, prior, posterior
  χ²(k)        = Σᵢ (yᵢ − k·xᵢ)² / σ²                       (misfit)
  likelihood   p(data | k) ∝ exp(−χ²(k) / 2)
  prior        p(k) = N(μₚ, σₚ²)   ∝ exp(−(k − μₚ)² / (2σₚ²))
  posterior    p(k | data) ∝ p(data | k)·p(k)
               −2 ln p(k | data) = χ²(k) + (k − μₚ)²/σₚ² + const

Metropolis algorithm (random walk)
  from the current k propose  k′ = k + s·N(0, 1)          (s = step size)
  accept k′ with probability  α = min(1, p(k′|data) / p(k|data))
                                = min(1, exp(−½[χ²(k′) − χ²(k)] − ½[(k′−μₚ)² − (k−μₚ)²]/σₚ²))
  i.e. draw u ~ U(0, 1): u < α → move to k′, otherwise stay at k (k is repeated).
  The histogram of the visited k (after the burn-in) approaches the posterior.

  Because the model is linear and prior and noise are Gaussian, the exact
  posterior is also Gaussian (shown dashed for comparison):
     1/σ_post² = Σx²/σ² + 1/σₚ² ,   k_post = σ_post²·(Σxy/σ² + μₚ/σₚ²).

Panels
  upper left  : data (with ±σ error bars), true line, current fit y = k·x and
                the last states of the chain (thin lines)
  upper right : the misfit χ²(k) (computed once when the data are initialised)
                and every visited state of the chain placed on it
  lower left  : controls
  lower right : prior, histogram of the samples (posterior) and the trace plot

How to use
  1. choose k_true, σ, N  → "Initialise data" (computes data and χ²(k))
  2. choose the starting guess k₀ and the prior (mean μₚ, standard deviation σₚ);
     changing them restarts the chain. A click in the χ² panel sets k₀.
  3. choose the step size and the batch size (1, 10, 100, 1000) and press
     "Run MCMC batch" repeatedly. With a batch of 1 the last proposal is shown
     (green = accepted, red = rejected) together with α and u.

Run from a terminal:      julia Introduction_MCMC_1.jl
or from the Julia REPL:   include("Introduction_MCMC_1.jl")
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
const STEP     = 0.05               # default proposal step size
const BATCHES  = [1, 10, 100, 1000] # batch sizes of the "Run" button
const NSPAG    = 40                 # number of last chain states drawn in the data panel

const C_DATA  = :black
const C_TRUE  = :gray45
const C_CUR   = :darkorange2        # current state of the chain
const C_CHAIN = :seagreen           # visited states / histogram
const C_REJ   = :red3               # rejected proposals
const C_PRIOR = :royalblue3
const C_POST  = :black              # exact posterior
const C_MIS   = :purple4            # misfit

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

"ln of the (unnormalised) Gaussian prior."
logprior(k, μ, s) = -0.5 * ((k - μ) / s)^2

"ln of the (unnormalised) posterior = −χ²/2 + ln prior."
logpost(k, S, μ, s) = -0.5 * chi2(k, S) + logprior(k, μ, s)

gauss(k, m, s) = exp(-0.5 * ((k - m) / s)^2) / (s * sqrt(2π))

"Exact Gaussian posterior (mean, standard deviation) of the linear model."
function exact_posterior(S::DataSums, μ, s)
    A = S.xx / S.s2 + 1 / s^2
    B = S.xy / S.s2 + μ / s^2
    return B / A, 1 / sqrt(A)
end

"Least-squares slope k̂ (minimum of χ²) and its standard error."
lsq(S::DataSums) = (S.xy / S.xx, sqrt(S.s2 / S.xx))

"""
One Metropolis step from k (with ln posterior lp). Returns the proposal, its ln
posterior, the acceptance probability α, the uniform number u and whether the
proposal was accepted.
"""
function metropolis_step(rng, k, lp, step, S, μ, s)
    kp  = k + step * randn(rng)
    lpp = logpost(kp, S, μ, s)
    α   = exp(min(0.0, lpp - lp))
    u   = rand(rng)
    return (kp = kp, lpp = lpp, α = α, u = u, acc = u < α)
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
    lps    = Float64[]                                # their ln posterior
    props  = Float64[]                                # all proposals
    accs   = Bool[]                                   # accepted?
    laststep = Ref{Any}(nothing)                      # last step (for batch size 1)
    lastn  = Ref(0)                                   # size of the last batch

    # ---------------------------------------------------------------- layout
    fig = Figure(size = (1440, 880), fontsize = 14)          # the layout adapts when the window is resized
    Label(fig[0, 1:2], "Introduction to MCMC  –  Bayesian fit of  y = k·x  with the Metropolis algorithm",
          fontsize = 20, font = :bold)
    left  = fig[1, 1] = GridLayout(valign = :top)
    right = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Fixed(540))

    axD = Axis(left[1, 1]; xlabel = "x", ylabel = "y")        # no fixed height: it takes the space the controls leave
    axM = Axis(right[1, 1]; xlabel = "k", ylabel = "χ²(k)",
               ytickformat = vs -> [@sprintf("%.3g", v) for v in vs])
    axP = Axis(right[2, 1]; xlabel = "k", ylabel = "probability density")
    axT = Axis(right[3, 1]; xlabel = "iteration", ylabel = "k", title = "trace plot")
    rowsize!(right, 1, Relative(0.40))
    rowgap!(right, 8)
    for a in (axD, axM, axP, axT), name in (:rectanglezoom, :scrollzoom, :dragpan)
        name in keys(interactions(a)) && deregister_interaction!(a, name)
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
    sdvals = filter(v -> 0.02 <= v <= 20, sort(unique(round.(vcat([m .* 10.0^e for e in -3:1 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3))))
    sgP = SliderGrid(left[6, 1],
        (label = "start k₀", range = KLIM[1]:0.05:KLIM[2], startvalue = K_START, format = "{:.2f}"),
        (label = "prior mean μₚ", range = KLIM[1]:0.05:KLIM[2], startvalue = PRIOR_MU, format = "{:.2f}"),
        (label = "prior st.dev. σₚ", range = sdvals, startvalue = PRIOR_SD, format = v -> @sprintf("%.3g  (σₚ² = %.3g)", v, v^2));
        tellheight = true, width = 500, halign = :left)
    sl_k0, sl_mu, sl_sp = sgP.sliders

    # --- 3. sampler
    Label(left[7, 1], "3 · Metropolis sampler   k′ = k + s·N(0,1)", font = :bold, halign = :left, tellwidth = false)
    stepvals = filter(v -> 0.001 <= v <= 3, sort(unique(round.(vcat([m .* 10.0^e for e in -3:1 for m in (1, 1.5, 2, 3, 5, 7)]), sigdigits = 3))))
    sgS = SliderGrid(left[8, 1],
        (label = "step size s", range = stepvals, startvalue = STEP, format = "{:.3g}"),
        (label = "batch size", range = BATCHES, startvalue = BATCHES[1], format = "{:d} samples"),
        (label = "burn-in", range = 0:10:2000, startvalue = 0, format = "discard first {:d}");
        tellheight = true, width = 500, halign = :left)
    sl_step, sl_batch, sl_burn = sgS.sliders
    bS = left[9, 1] = GridLayout(halign = :left)
    btn_run   = Button(bS[1, 1], label = "Run MCMC batch", buttoncolor = RGBf(0.80, 0.95, 0.80), width = 220)
    btn_reset = Button(bS[1, 2], label = "Reset chain")

    tg = left[10, 1] = GridLayout(halign = :left)
    tg_zoom  = Toggle(tg[1, 1], active = false); Label(tg[1, 2], "zoom on the posterior", halign = :left)
    tg_log   = Toggle(tg[1, 3], active = false); Label(tg[1, 4], "log scale for χ²", halign = :left)
    tg_exact = Toggle(tg[2, 1], active = true);  Label(tg[2, 2], "exact posterior", halign = :left)
    tg_rej   = Toggle(tg[2, 3], active = true);  Label(tg[2, 4], "rejected proposals", halign = :left)
    colgap!(tg, 8); rowgap!(tg, 4)

    # fixed heights (5 and 2 lines), so that the data axis does not jump when the text changes
    info   = Label(left[11, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   fontsize = 13, height = 80)
    status = Label(left[12, 1], " ", halign = :left, valign = :top, tellwidth = false, justification = :left,
                   word_wrap = true, color = :gray20, fontsize = 13, height = 34)
    rowgap!(left, 4)
    for sg in (sgD, sgP, sgS)
        rowgap!(sg.layout, 2)
    end

    # ------------------------------------------------------ plot objects
    P0 = Point2f[(0, 0)]
    # data panel
    o_dat  = Observable(P0)
    o_err  = Observable(Point2f[(0, 0), (0, 0)])
    o_true = Observable(Point2f[(0, 0), (1, 0)])
    o_spag = Observable(Point2f[(0, 0), (0, 0)])
    o_cur  = Observable(Point2f[(0, 0), (1, 0)])
    o_prop = Observable(Point2f[(0, 0), (1, 0)]); c_prop = Observable{Any}(C_CHAIN); vis_prop = Observable(false)
    # misfit panel
    o_mis  = Observable(P0); o_mpo = Observable(P0)
    o_vis  = Observable(P0); o_rej = Observable(P0)
    o_mcur = Observable(P0); o_mprop = Observable(P0)
    o_arrow = Observable(Point2f[(0, 0), (0, 0)])
    o_kt   = Observable(K_TRUE); o_khat = Observable(K_TRUE)
    # posterior panel
    o_prior = Observable(P0); o_exact = Observable(P0)
    o_hist  = Observable(P0); o_bw = Observable(1.0); vis_hist = Observable(false)
    # trace panel
    o_trace = Observable(P0); o_burn = Observable(Rect2f(0, 0, 0, 1)); o_pm = Observable(0.0)
    o_band  = Observable(Rect2f(0, 0, 1, 1))

    # upper left: data
    lines!(axD, o_true; color = C_TRUE, linewidth = 2, linestyle = :dash, label = "true line  y = k_true·x")
    linesegments!(axD, o_spag; color = (C_CHAIN, 0.25), linewidth = 1, label = "last $NSPAG states of the chain")
    lines!(axD, o_prop; color = c_prop, linewidth = 2, linestyle = :dot, visible = vis_prop, label = "proposal k′")
    lines!(axD, o_cur; color = C_CUR, linewidth = 3, label = "current fit  y = k·x")
    linesegments!(axD, o_err; color = (C_DATA, 0.6), linewidth = 1.2)
    scatter!(axD, o_dat; color = C_DATA, markersize = 9, label = "data ± σ")
    axislegend(axD; position = :lt, labelsize = 11, rowgap = 0, colgap = 6, nbanks = 2, patchsize = (16, 10),
               padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

    # upper right: misfit
    vlines!(axM, o_kt; color = C_TRUE, linestyle = :dash, linewidth = 1.5, label = "k_true")
    vlines!(axM, o_khat; color = C_MIS, linestyle = :dot, linewidth = 1.5, label = "least squares k̂ (min χ²)")
    lines!(axM, o_mpo; color = (C_PRIOR, 0.8), linewidth = 2, linestyle = :dash,
           label = "χ² + (k−μₚ)²/σₚ²  = −2 ln posterior + const")
    lines!(axM, o_mis; color = C_MIS, linewidth = 3, label = "misfit χ²(k)")
    scatter!(axM, o_rej; color = (C_REJ, 0.35), marker = :xcross, markersize = 7, visible = tg_rej.active,
             label = "rejected proposals")
    scatter!(axM, o_vis; color = (C_CHAIN, 0.45), markersize = 7, label = "visited states of the chain")
    lines!(axM, o_arrow; color = c_prop, linewidth = 2, visible = vis_prop)
    scatter!(axM, o_mprop; color = c_prop, marker = :diamond, markersize = 15, strokecolor = :black, strokewidth = 1,
             visible = vis_prop)
    scatter!(axM, o_mcur; color = C_CUR, markersize = 18, strokecolor = :black, strokewidth = 1.5,
             label = "current state")
    axislegend(axM; position = :ct, labelsize = 11, rowgap = 0, padding = (6, 6, 4, 4), backgroundcolor = (:white, 0.85))

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
        μ, s = prior_pars()
        empty!(chain); empty!(lps); empty!(props); empty!(accs)
        k0 = sl_k0.value[]
        push!(chain, k0); push!(lps, logpost(k0, S[], μ, s))
        laststep[] = nothing; lastn[] = 0
    end

    "Run n Metropolis steps."
    function run_batch!(n)
        μ, s = prior_pars(); step = sl_step.value[]
        k, lp = chain[end], lps[end]
        for _ in 1:n
            r = metropolis_step(rng_c, k, lp, step, S[], μ, s)
            push!(props, r.kp); push!(accs, r.acc)
            if r.acc
                k, lp = r.kp, r.lpp
            end
            push!(chain, k); push!(lps, lp)
            laststep[] = merge((k_old = chain[end-1],), r)
        end
        lastn[] = n
    end

    "k window of the k axes (full range or zoomed on the posterior)."
    function kwindow()
        μ, s = prior_pars()
        m, sd = exact_posterior(S[], μ, s)
        tg_zoom.active[] || return KLIM
        return (m - 6sd, m + 6sd)
    end

    "Everything that depends on the prior (curves only)."
    function draw_prior!()
        μ, s = prior_pars()
        m, sd = exact_posterior(S[], μ, s)
        o_mpo[]   = Point2f.(kgrid, chi2.(kgrid, Ref(S[])) .+ ((kgrid .- μ) ./ s) .^ 2)
        k1, k2 = kwindow()
        kk = collect(range(min(k1, KLIM[1]), max(k2, KLIM[2]), length = 4001))
        kk = sort(vcat(kk, range(m - 6sd, m + 6sd, length = 801)))   # resolve a narrow posterior
        o_prior[] = Point2f.(kk, gauss.(kk, μ, s))
        o_exact[] = Point2f.(kk, gauss.(kk, m, sd))
        o_pm[]    = m
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
        # misfit panel
        o_vis[]  = Point2f.(chain, chi2.(chain, Ref(Sd)))
        rej = props[.!accs]
        o_rej[]  = isempty(rej) ? Point2f[(NaN, NaN)] : Point2f.(rej, chi2.(rej, Ref(Sd)))
        o_mcur[] = [Point2f(kc, chi2(kc, Sd))]
        L = laststep[]
        show1 = lastn[] == 1 && L !== nothing
        vis_prop[] = show1
        if show1
            c_prop[]  = L.acc ? :forestgreen : C_REJ
            o_prop[]  = [Point2f(XLIM[1], L.kp * XLIM[1]), Point2f(XLIM[2], L.kp * XLIM[2])]
            o_mprop[] = [Point2f(L.kp, chi2(L.kp, Sd))]
            o_arrow[] = [Point2f(L.k_old, chi2(L.k_old, Sd)), Point2f(L.kp, chi2(L.kp, Sd))]
        end
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
        txt = @sprintf("iterations: %d     accepted: %d  (%.0f %%)     current k = %s,  χ² = %.4g",
                       n, nacc, n == 0 ? 0.0 : 100nacc / n, fmt(kc), chi2(kc, Sd))
        if show1
            txt *= @sprintf("\nlast step: k = %s → proposal k′ = %s   χ²: %.4g → %.4g", fmt(L.k_old), fmt(L.kp),
                            chi2(L.k_old, Sd), chi2(L.kp, Sd))
            txt *= @sprintf("\n   α = min(1, p(k′|d)/p(k|d)) = %.3g,   u = %.3g   →  %s", L.α, L.u,
                            L.acc ? "u < α: ACCEPTED (move to k′)" : "u ≥ α: REJECTED (stay at k, k repeated)")
        end
        txt *= isempty(kept) ? "\nsamples after burn-in: none yet" :
               @sprintf("\nsamples after burn-in: %d     mean = %s,  st.dev. = %s", length(kept),
                        fmt(sum(kept) / length(kept)),
                        fmt(length(kept) > 1 ? sqrt(sum(abs2, kept .- sum(kept) / length(kept)) / (length(kept) - 1)) : 0.0))
        txt *= @sprintf("\nexact posterior: %s ± %s     least squares k̂ = %s ± %s", fmt(m), fmt(sd), fmt(kh), fmt(kse))
        info.text[] = txt
        axP.title = "prior and posterior  (" * (isempty(kept) ? "no samples yet" : "$(length(kept)) samples") * ")"
        axM.title = "misfit χ²(k) = Σ (yᵢ − k·xᵢ)²/σ²  –  click to set the starting guess k₀"
    end

    "Axis limits (k window shared by the three k axes)."
    function draw_limits!()
        k1, k2 = kwindow()
        Sd = S[]; μ, s = prior_pars()
        # misfit
        ks = range(k1, k2, length = 401)
        cv = chi2.(ks, Ref(Sd))
        cmin = minimum(cv); cmax = maximum(cv)
        if tg_log.active[]
            lo = max(cmin, 1e-3) / 2; hi = cmax * 3
            limits!(axM, k1, k2, lo, hi)               # positive limits first, then the scale
            axM.yscale = log10
        else
            axM.yscale = identity
            limits!(axM, k1, k2, 0, cmax * 1.12 + 1e-9)
        end
        # posterior
        m, sd = exact_posterior(Sd, μ, s)
        ks2 = range(k1, k2, length = 2001)
        ycurve = max(maximum(gauss.(ks2, μ, s)), tg_exact.active[] ? gauss(m, m, sd) : 0.0)
        yhist  = vis_hist[] ? maximum(p[2] for p in o_hist[]) : 0.0
        ymax   = max(ycurve, min(yhist, 2.5ycurve))           # a few samples give a very tall histogram: clip it
        limits!(axP, k1, k2, 0, 1.15 * ymax + 1e-9)
        # trace
        n = length(chain) - 1
        limits!(axT, -0.02 * max(n, 10), max(n, 10) * 1.02, k1, k2)
        o_burn[] = Rect2f(0, k1, min(sl_burn.value[], n), k2 - k1)
        o_band[] = Rect2f(0, m - sd, max(n, 10) * 1.02, 2sd)
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
        set_status!("Data initialised and χ²(k) computed. Choose k₀ and the prior, then press \"Run MCMC batch\".")
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
    on(_ -> draw_chain!(), sl_burn.value)
    on(_ -> (draw_prior!(); draw_chain!()), tg_zoom.active)
    on(_ -> draw_limits!(), tg_log.active)
    on(_ -> draw_limits!(), tg_exact.active)
    register_interaction!(axM, :pick_k0) do event::MouseEvent, axis
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
                            step = sl_step, batch = sl_batch, burn = sl_burn),
                 toggles = (zoom = tg_zoom, log = tg_log, exact = tg_exact, rej = tg_rej),
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
