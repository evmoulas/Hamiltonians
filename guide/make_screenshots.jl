#=
make_screenshots.jl – regenerate the annotated screenshots used in guide.tex

    julia make_screenshots.jl mcmc1     # one program: mcmc1, mcmc2, mcmc3, hmc1, hmc2, cmp, cmp2
    julia make_screenshots.jl all       # all programs, each in its own Julia process

Every program defines the same global names, so each one is loaded in a separate
Julia process. The program builds the GUI without showing it, puts it into a
useful state, draws numbered red markers on the widgets (positions are taken
from the widgets themselves) and saves the picture to figures/. The text shown
in the program (info line, table) is printed, so that the captions can quote it.
=#

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIGS = joinpath(@__DIR__, "figures")
const PROGRAMS = Dict("mcmc1" => "Introduction_MCMC_1.jl", "mcmc2" => "Introduction_MCMC_2.jl",
                      "mcmc3" => "Introduction_MCMC_3.jl", "hmc1" => "Introduction_HMC_1.jl",
                      "hmc2" => "Introduction_HMC_2.jl", "cmp" => "Comparison_MCMC_to_HMC_1.jl",
                      "cmp2" => "Comparison_MCMC_to_HMC_2.jl")

if length(ARGS) == 1 && ARGS[1] == "all"
    for k in ("mcmc1", "mcmc2", "mcmc3", "hmc1", "hmc2", "cmp", "cmp2")
        run(`$(Base.julia_cmd()) $(@__FILE__) $k`)
    end
    exit()
end
const EX = ARGS[1]
mkpath(FIGS)

include(joinpath(ROOT, PROGRAMS[EX]))              # defines build_app(), loads GLMakie

# ----------------------------------------------------------------------------
# helpers (pixel coordinates of the figure, origin bottom left)
# ----------------------------------------------------------------------------
const MARK = RGBf(0.85, 0.05, 0.05)

blocks(fig, T) = [b for b in fig.content if b isa T]
txt(b) = (s = b isa Button ? b.label[] : b isa Union{Axis,Axis3} ? b.title[] : b.text[];
          s isa AbstractString ? s : string(s))
"Block of type T whose text equals s (or, if none, contains s)."
function findblock(fig, T, s)
    bs = blocks(fig, T)
    for b in bs
        txt(b) == s && return b
    end
    for b in bs
        occursin(s, txt(b)) && return b
    end
    error("no $T containing \"$s\"")
end
bbox(b) = b.layoutobservables.computedbbox[]
axbox(ax) = (v = ax.scene.viewport[]; Rect2f(v.origin..., v.widths...))
function union_rect(rs...)
    lo = reduce((a, b) -> min.(a, b), [r.origin for r in rs])
    hi = reduce((a, b) -> max.(a, b), [r.origin .+ r.widths for r in rs])
    Rect2f(lo..., (hi .- lo)...)
end
function data2px(ax, x, y)
    v = ax.scene.viewport[]; l = ax.finallimits[]
    Point2f(v.origin .+ (Point2f(x, y) .- l.origin) ./ l.widths .* v.widths)
end
cy(b) = (r = bbox(b); r.origin[2] + r.widths[2] / 2)

overlay(fig) = Scene(fig.scene; camera = campixel!, clear = false)

function badge!(sc, p, n)
    s = scatter!(sc, [p]; markersize = 32, color = MARK, strokecolor = :white, strokewidth = 2.5)
    t = text!(sc, [p]; text = string(n), color = :white, font = :bold, fontsize = 18, align = (:center, :center))
    translate!(s, 0, 0, 20); translate!(t, 0, 0, 21)
end
"Red frame around a rectangle and a numbered badge at one of its corners."
function mark!(sc, r::Rect2, n; pad = 3, corner = :lt)
    r2 = Rect2f((r.origin .- pad)..., (r.widths .+ 2pad)...)
    o, w = r2.origin, r2.widths
    lines!(sc, Point2f[o, o .+ (w[1], 0), o .+ w, o .+ (0, w[2]), o]; color = MARK, linewidth = 2.5)
    p = corner === :lt ? Point2f(o[1] - 16, o[2] + w[2]) : corner === :rt ? Point2f(o[1] + w[1] + 16, o[2] + w[2]) :
        corner === :lb ? Point2f(o[1] - 16, o[2]) : Point2f(o[1] + w[1] + 16, o[2])
    fw, fh = sc.viewport[].widths
    badge!(sc, Point2f(clamp(p[1], 16, fw - 16), clamp(p[2], 16, fh - 16)), n)
end
mark!(sc, b, n; kw...) = mark!(sc, bbox(b), n; kw...)
"Mark a click position inside an axis: ring + badge."
function click!(sc, ax, x, y, n)
    p = data2px(ax, x, y)
    scatter!(sc, [p]; marker = :circle, markersize = 28, color = :transparent, strokecolor = MARK, strokewidth = 3.5)
    badge!(sc, p .+ Point2f(24, 24), n)
end
"Slider row of a SliderGrid: the label, the slider on the same height and the value label."
function sliderrow(fig, s)
    for exact in (true, false), sg in blocks(fig, SliderGrid), (i, lab) in enumerate(sg.labels)
        t = txt(lab)
        (exact ? t == s : occursin(s, t)) || continue
        return union_rect(bbox(lab), bbox(sg.sliders[i]), bbox(sg.valuelabels[i]))
    end
    error("no slider labelled \"$s\"")
end
rows(fig, ss...) = union_rect([sliderrow(fig, s) for s in ss]...)
"Toggle together with the label to its right."
function with_label(fig, s)
    lab = findblock(fig, Label, s)
    lb  = bbox(lab)
    cands = [b for b in blocks(fig, Toggle) if bbox(b).origin[1] < lb.origin[1]]
    t = argmin(b -> abs(cy(b) - cy(lab)) + 0.2 * (lb.origin[1] - bbox(b).origin[1]), cands)
    union_rect(bbox(t), lb)
end
btn(fig, s) = bbox(findblock(fig, Button, s))

"Finish the animations of toggles that were switched from the program (no render loop off screen)."
function settle!(fig)
    ev = events(fig.scene); t = ev.tick[]
    ev.tick[] = Makie.Tick(Makie.OneTimeRenderTick, t.count + 1, t.time + 10.0, 10.0)
end

function shot(fig, name; annotate = sc -> nothing, width = 1.4)
    settle!(fig)
    Makie.update_state_before_display!(fig)
    sc = overlay(fig)
    annotate(sc)
    save(joinpath(FIGS, name), fig; px_per_unit = width)
    empty!(sc)
    println("\n=== saved ", name)
    for b in blocks(fig, Label)                         # the info text of the program
        t = txt(b)
        (startswith(t, "iterations") || startswith(t, "trajectory") || startswith(t, "HMC traj") ||
         startswith(t, "L·ε")) && println(t)
    end
end
setv!(sl, v) = set_close_to!(sl, v)

# ----------------------------------------------------------------------------
# the screenshots of every program
# ----------------------------------------------------------------------------
if EX == "mcmc1"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axM = findblock(fig, Axis, "misfit")
    app.run(1); app.run(1); app.run(1)
    shot(fig, "mcmc1_overview.png"; annotate = sc -> begin
        mark!(sc, rows(fig, "k_true", "noise σ", "N points"), 1)
        mark!(sc, btn(fig, "Initialise data (compute χ²)"), 2)
        mark!(sc, rows(fig, "start k₀", "prior mean μₚ", "prior st.dev. σₚ"), 3)
        click!(sc, axM, 0.3, 230, 3)
        mark!(sc, rows(fig, "step size s"), 4)
        mark!(sc, rows(fig, "batch size"), 5; corner = :rt)
        mark!(sc, btn(fig, "Run MCMC batch"), 6)
        mark!(sc, btn(fig, "Reset chain"), 7; corner = :rt)
        mark!(sc, rows(fig, "burn-in"), 8; corner = :rb)
        mark!(sc, union_rect(with_label(fig, "zoom on the posterior"), with_label(fig, "rejected proposals")), 9)
    end)
    setv!(s.batch, 100); for _ in 1:20; app.run(100); end; setv!(s.burn, 100)
    shot(fig, "mcmc1_converged.png")
    t.zoom.active[] = true; t.log.active[] = true
    shot(fig, "mcmc1_zoom_log.png")
    t.log.active[] = false
    setv!(s.burn, 100); setv!(s.step, 0.01); app.reset(); for _ in 1:10; app.run(100); end
    shot(fig, "mcmc1_step_small.png")
    setv!(s.step, 2.0); app.reset(); for _ in 1:10; app.run(100); end
    shot(fig, "mcmc1_step_large.png")
    setv!(s.step, 0.2); app.reset(); for _ in 1:10; app.run(100); end
    shot(fig, "mcmc1_step_good.png")

elseif EX == "mcmc2"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axP = findblock(fig, Axis, "prior, posterior")
    setv!(s.k0, 1.1); t.zoom.active[] = true
    app.run(1); app.run(1)
    shot(fig, "mcmc2_overview.png"; annotate = sc -> begin
        mark!(sc, bbox(only(blocks(fig, Menu))), 1; corner = :rt)
        mark!(sc, rows(fig, "width s", "drift d"), 2)
        mark!(sc, with_label(fig, "Hastings correction"), 3; corner = :rt)
        mark!(sc, union_rect(btn(fig, "Run MCMC batch"), btn(fig, "Reset chain")), 4)
        mark!(sc, axbox(axP), 5; corner = :rt)
        mark!(sc, union_rect(with_label(fig, "zoom on posterior"), with_label(fig, "exact posterior"),
                             with_label(fig, "proposal density q")), 6)
    end)
    setv!(s.k0, 1.5); setv!(s.step, 0.1); setv!(s.batch, 100); setv!(s.burn, 100)
    app.reset(); for _ in 1:50; app.run(100); end
    shot(fig, "mcmc2_drift_on.png")
    t.hastings.active[] = false; app.reset(); for _ in 1:50; app.run(100); end
    shot(fig, "mcmc2_drift_off.png")
    t.hastings.active[] = true
    app.proposal(4); setv!(s.step, 0.15); setv!(s.burn, 0); setv!(s.k0, 1.1); app.reset(); app.run(1)
    shot(fig, "mcmc2_mala_step.png")
    app.proposal(3); setv!(s.step, 0.15); setv!(s.shift, 1.3); setv!(s.k0, 1.5); setv!(s.burn, 100)
    t.hastings.active[] = false; app.reset(); for _ in 1:50; app.run(100); end
    shot(fig, "mcmc2_indep_off.png")

elseif EX == "mcmc3"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axM = findblock(fig, Axis, "misfit")
    setv!(s.batch, 100); for _ in 1:20; app.run(100); end
    shot(fig, "mcmc3_overview.png"; annotate = sc -> begin
        mark!(sc, union_rect(rows(fig, "k_true", "a_true", "noise σ", "N points"), btn(fig, "Initialise data (compute χ² map)")), 1)
        mark!(sc, rows(fig, "prior mean μ_k", "prior st.dev. σ_k", "prior mean μ_a", "prior st.dev. σ_a"), 2)
        click!(sc, axM, -0.5, -5.0, 3)
        mark!(sc, union_rect(bbox(only(blocks(fig, Menu))), with_label(fig, "Hastings\ncorrection")), 4)
        mark!(sc, rows(fig, "step size s", "batch size", "burn-in"), 5)
        mark!(sc, union_rect(btn(fig, "Run MCMC batch"), btn(fig, "Reset chain")), 6)
        mark!(sc, btn(fig, "learn mean, Σ from chain"), 7; corner = :rb)
        mark!(sc, union_rect(with_label(fig, "zoom on posterior"), with_label(fig, "chain path")), 8)
    end)
    t.zoom.active[] = true
    app.proposal(2); setv!(s.step, 0.3); app.setstart((1.0, 5.5)); for _ in 1:10; app.run(100); end
    shot(fig, "mcmc3_one_at_a_time.png")
    app.proposal(1); setv!(s.step, 0.3); app.reset(); for _ in 1:30; app.run(100); end; app.learn()
    app.proposal(3); setv!(s.step, 1.5); app.setstart((1.0, 5.5)); for _ in 1:10; app.run(100); end
    shot(fig, "mcmc3_learned.png")
    t.zoom.active[] = false
    app.proposal(4); setv!(s.step, 0.3); app.setstart((-0.5, -5.0)); for _ in 1:10; app.run(100); end; app.run(1)
    shot(fig, "mcmc3_indep_stuck.png")

elseif EX == "hmc1"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axH = findblock(fig, Axis, "phase space")
    for _ in 1:5; app.run(1); end
    shot(fig, "hmc1_overview.png"; annotate = sc -> begin
        mark!(sc, union_rect(rows(fig, "k_true", "noise σ", "N points"), btn(fig, "Initialise data (compute χ²)")), 1)
        mark!(sc, rows(fig, "start k₀", "prior mean μₚ", "prior st.dev. σₚ"), 2)
        mark!(sc, rows(fig, "leapfrog step ε", "leapfrog steps L", "mass M"), 3)
        mark!(sc, rows(fig, "batch size", "burn-in"), 4; corner = :rt)
        mark!(sc, union_rect(btn(fig, "Run HMC batch"), btn(fig, "Reset chain")), 5)
        mark!(sc, bbox(findblock(fig, Label, "trajectory =")), 6; corner = :rt)
        mark!(sc, axbox(axH), 7; corner = :rt)
        mark!(sc, union_rect(with_label(fig, "zoom on posterior"), with_label(fig, "exact posterior"),
                             with_label(fig, "contours of H")), 8)
    end)
    setv!(s.batch, 100); for _ in 1:10; app.run(100); end; t.zoom.active[] = true; setv!(s.burn, 20); app.run(1)
    shot(fig, "hmc1_converged.png")
    setv!(s.L, 33); setv!(s.k0, 1.3); setv!(s.batch, 100); for _ in 1:3; app.run(100); end; app.run(1)
    shot(fig, "hmc1_uturn.png")
    setv!(s.L, 8); app.reset(); for _ in 1:3; app.run(100); end; app.run(1)
    shot(fig, "hmc1_quarter.png")
    t.zoom.active[] = false; setv!(s.L, 10); setv!(s.eps, 0.3); setv!(s.k0, 1.0)
    app.reset(); for _ in 1:4; app.run(1); end
    shot(fig, "hmc1_unstable.png")

elseif EX == "hmc2"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axM = findblock(fig, Axis, "misfit"); axPk = findblock(fig, Axis, "phase space (k, p_k)")
    axPa = findblock(fig, Axis, "phase space (a, p_a)")
    for _ in 1:6; app.run(1); end
    shot(fig, "hmc2_overview.png"; annotate = sc -> begin
        mark!(sc, union_rect(bbox(only(blocks(fig, Menu))), with_label(fig, "jitter")), 1; corner = :rt)
        mark!(sc, rows(fig, "leapfrog step ε", "leapfrog steps L", "batch size", "burn-in"), 2)
        mark!(sc, union_rect(btn(fig, "Run HMC batch"), btn(fig, "Reset chain")), 3)
        mark!(sc, btn(fig, "learn Σ from chain"), 4; corner = :rt)
        mark!(sc, bbox(findblock(fig, Label, "trajectory =")), 5; corner = :lb)
        mark!(sc, axbox(axM), 6)
        mark!(sc, union_rect(axbox(axPk), axbox(axPa)), 7; corner = :rt)
        mark!(sc, union_rect(with_label(fig, "zoom on posterior"), with_label(fig, "visited states")), 8)
    end)
    setv!(s.eps, 0.15); setv!(s.batch, 100); app.reset(); for _ in 1:10; app.run(100); end; app.run(1)
    shot(fig, "hmc2_resonance.png")
    t.jitter.active[] = true; setv!(s.burn, 20); app.reset(); for _ in 1:10; app.run(100); end; app.run(1)
    shot(fig, "hmc2_jitter.png")
    t.jitter.active[] = false; setv!(s.burn, 0); setv!(s.eps, 0.05); app.reset(); for _ in 1:30; app.run(100); end; app.learn()
    app.mass(3); setv!(s.eps, 0.3); setv!(s.L, 5); app.reset(); for _ in 1:5; app.run(100); end
    t.zoom.active[] = true; app.run(1)
    shot(fig, "hmc2_fullmass.png")

elseif EX == "cmp"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axE = findblock(fig, Axis, "convergence"); axT = findblock(fig, Axis, "trace of k")
    axs = [findblock(fig, Axis, "Metropolis (random walk)"), findblock(fig, Axis, "Hamiltonian Monte Carlo")]
    for _ in 1:20; app.run(100); end
    printtable() = foreach(i -> println(rpad(i, 3), app.cells[i, 1].text[], "   |   ", app.cells[i, 2].text[]), 1:size(app.cells, 1))
    shot(fig, "cmp_overview.png"; annotate = sc -> begin
        mark!(sc, union_rect(rows(fig, "k_true", "a_true", "noise σ", "N points"), btn(fig, "Initialise data")), 1)
        mark!(sc, rows(fig, "Metropolis step s", "HMC leapfrog step ε", "HMC leapfrog steps L"), 2)
        mark!(sc, union_rect(with_label(fig, "precondition"), with_label(fig, "jitter")), 3)
        mark!(sc, rows(fig, "iterations per run", "burn-in"), 4)
        mark!(sc, union_rect(btn(fig, "Run both"), btn(fig, "Reset"), with_label(fig, "equal cost")), 5)
        mark!(sc, union_rect(axbox(axs[1]), axbox(axs[2])), 6)
        mark!(sc, axbox(axT), 7)
        mark!(sc, axbox(axE), 8)
        mark!(sc, union_rect(bbox(findblock(fig, Label, "Metropolis")), bbox(findblock(fig, Label, "HMC")),
                             bbox(findblock(fig, Label, "iterations")), bbox(findblock(fig, Label, "exact posterior:"))), 9)
        mark!(sc, with_label(fig, "x-axis: model evaluations"), 10)
    end); printtable()
    t.cost.active[] = true
    shot(fig, "cmp_cost.png"); printtable()
    t.cost.active[] = false
    t.pre.active[] = true; setv!(s.s, 1.5); setv!(s.eps, 0.3); setv!(s.L, 5); t.eq.active[] = true
    setv!(s.burn, 100); t.zoom.active[] = true
    app.reset(); for _ in 1:10; app.run(100); end
    shot(fig, "cmp_precond.png"); printtable()
    t.cost.active[] = true
    shot(fig, "cmp_precond_cost.png"); printtable()

elseif EX == "cmp2"
    fig, app = build_app(); s = app.sliders; t = app.toggles
    axP = findblock(fig, Axis, "chains projected"); axS = findblock(fig, Axis, "scan")
    axE = findblock(fig, Axis, "convergence")
    function report(name)
        r = app.scanres[]
        if r !== nothing
            for D in 1:DMAX
                @printf("  D=%2d  MH %8.3f [%7.3f, %7.3f] per run %7.1f   HMC %8.3f [%7.3f, %7.3f] per run %7.1f\n", D,
                        r[:mh][D, :]..., r[:hmc][D, :]...)
            end
        end
        foreach(i -> println("  ", app.cells[i, 1].text[], "   |   ", app.cells[i, 2].text[]), 1:size(app.cells, 1))
        T = app.tune[]; println("  s = ", T.s, "  ε = ", T.ε, "  L = ", T.L, "  ωmax/ωmin = ", T.ωmax / T.ωmin)
    end
    app.scan(); for _ in 1:20; app.run(100); end
    shot(fig, "cmp2_overview.png"; annotate = sc -> begin
        mark!(sc, union_rect(bbox(only(blocks(fig, Menu))), rows(fig, "parameters D", "noise σ", "N points", "show coefficient j"),
                             btn(fig, "Initialise data")), 1)
        mark!(sc, union_rect(with_label(fig, "precondition both"), with_label(fig, "jitter ε")), 2)
        mark!(sc, rows(fig, "Metropolis step s", "HMC leapfrog step ε", "HMC leapfrog steps L"), 3)
        mark!(sc, rows(fig, "HMC iterations per run", "burn-in", "scan budget"), 4)
        mark!(sc, union_rect(btn(fig, "Run both"), btn(fig, "Reset"), with_label(fig, "equal cost")), 5)
        mark!(sc, btn(fig, "Scan D = 1 … 10"), 6; corner = :rt)
        mark!(sc, axbox(axP), 7)
        mark!(sc, axbox(axS), 8; corner = :rt)
        mark!(sc, axbox(axE), 9)
        mark!(sc, union_rect(bbox(findblock(fig, Label, "Metropolis")), bbox(findblock(fig, Label, "HMC")),
                             bbox(findblock(fig, Label, "iterations")), bbox(findblock(fig, Label, "ω_max/ω_min"))), 10)
        mark!(sc, with_label(fig, "x-axis: model evaluations"), 11)
    end); report("overview")
    setv!(s.D, 10); for _ in 1:20; app.run(100); end
    shot(fig, "cmp2_D10.png"); report("D10")
    t.pre.active[] = false; app.scan(); for _ in 1:20; app.run(100); end
    shot(fig, "cmp2_legendre_noprecond.png"); report("legendre no precond")
    app.menu.i_selected[] = 2; app.scan(); for _ in 1:20; app.run(100); end
    shot(fig, "cmp2_monomial_noprecond.png"); report("monomial no precond")
end
