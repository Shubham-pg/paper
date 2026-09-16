

"""
exp5_ablation.py -> Table "Ablation results" (A1, A3) + A6 parameter sensitivity

A1  closure on/off
A3  climber only / diver only / both
A6  sweep tau, Delta, K, alpha -> components, |V|/|E|, query time, min clearance

(A2 lives in exp2, A4 in exp3, A5 in exp4.)

Usage:  python exp5_ablation.py [--Q 40]
"""
import argparse, json, math, time
import numpy as np
import networkx as nx
import tc_core as tc



def inject_single(p0, F, skel, direction, alpha=None, max_steps=4000,
                  contact=2.5, dcage=None, max_coast=25):
    """One-sided scout: direction=+1 climber, -1 diver.
    Mirrors tc.inject_gradient exactly, minus the bidirectional loop."""
    alpha = tc.PARAMS['alpha_step'] if alpha is None else alpha
    H, W = skel.shape
    if dcage is None:
        dcage = tc.cage_distance(skel)
    cy, cx = float(p0[0]), float(p0[1])
    ly = lx = 0.0
    coast = 0
    for k in range(max_steps):
        y = int(cy); x = int(cx)
        if not (0 <= y < H and 0 <= x < W):
            return None, k, False

        if dcage[y, x] <= contact:
            y0 = max(0, y - 4); y1 = min(H, y + 5)
            x0 = max(0, x - 4); x1 = min(W, x + 5)
            sub = np.argwhere(skel[y0:y1, x0:x1] > 0)
            if len(sub):
                d = np.hypot(sub[:, 0] + y0 - y, sub[:, 1] + x0 - x)
                j = int(d.argmin())
                return (y0 + int(sub[j, 0]), x0 + int(sub[j, 1])), k + 1, True

        m = F.mag[y, x]
        if m < 0.05:
            if (ly == 0.0 and lx == 0.0) or coast >= max_coast:
                return None, k, False
            coast += 1
            cy += ly * alpha; cx += lx * alpha
            continue

        coast = 0
        ly = direction * F.gy[y, x] / m
        lx = direction * F.gx[y, x] / m
        cy += ly * alpha
        cx += lx * alpha
    return None, max_steps, False




def route(F, skel, G0, s, g, injector, K=None, alpha=None):
    t0 = time.perf_counter()
    hs = injector(s); hg = injector(g)
    if hs is None or hg is None:
        return None, (time.perf_counter() - t0) * 1e3, 0
    G = G0.copy()
    ns = tc.attach(G, F, skel, hs, 's'); ng = tc.attach(G, F, skel, hg, 'g')
    if ns is None or ng is None:
        return None, (time.perf_counter() - t0) * 1e3, 0
    paths = tc.enumerate_paths(G, ns, ng, K=K)
    if not paths:
        return None, (time.perf_counter() - t0) * 1e3, 0
    cands = [{'pixels': tc.node_path_to_pixels(G, p)} for p in paths]
    best, _ = tc.select_path(cands, F, 'lex')
    return best, (time.perf_counter() - t0) * 1e3, len(paths)


def batch(F, skel, G0, pairs, injector, K=None):
    ok, ms, clr, ks = 0, [], [], []
    for (s, g) in pairs:
        b, t, k = route(F, skel, G0, s, g, injector, K=K)
        ms.append(t); ks.append(k)
        if b:
            ok += 1; clr.append(b['min_clr'])
    return dict(feas=100.0 * ok / len(pairs),
                ms=float(np.mean(ms)),
                min_clr=float(np.mean(clr)) if clr else float('nan'),
                K=float(np.mean(ks)))


def attach_rates(F, skel, terms, dcage):
    """Pure attachment success per scout branch, and for the pair."""
    c = d = both = 0
    cs, ds = [], []
    for p in terms:
        hc, kc, okc = inject_single(p, F, skel, +1, dcage=dcage)
        hd, kd, okd = inject_single(p, F, skel, -1, dcage=dcage)
        c += okc; d += okd; both += (okc or okd)
        if okc: cs.append(kc)
        if okd: ds.append(kd)
    n = len(terms)
    return dict(climber=100.0 * c / n, diver=100.0 * d / n,
                both=100.0 * both / n,
                climber_steps=float(np.mean(cs)) if cs else float('nan'),
                diver_steps=float(np.mean(ds)) if ds else float('nan'))




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=int, default=40)
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    ap.add_argument('--out', default='results_exp5.json')
    a = ap.parse_args()

    R = {}
    for env in a.envs.split(','):
        print(f'\n===== {env} =====', flush=True)
        mask = tc.make_env(env)
        F = tc.Fields(mask)
        rng = np.random.default_rng(31)
        free = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)
        pairs = []
        while len(pairs) < a.Q:
            s = tuple(free[rng.integers(len(free))])
            g = tuple(free[rng.integers(len(free))])
            if math.dist(s, g) > 120:
                pairs.append((s, g))

        cage = tc.build_cage(F); skel = cage['skel']
        G0, _ = tc.build_graph(skel, F)
        raw = tc.build_cage(F, use_closure=False)
        Graw, _ = tc.build_graph(raw['skel'], F)

        E = {}
        grad = lambda p, sk=skel: tc.inject_gradient(p, F, sk)[1]
        E['full'] = batch(F, skel, G0, pairs, grad)
        E['A1_no_closure'] = batch(F, raw['skel'], Graw, pairs,
                                   lambda p: tc.inject_gradient(p, F, raw['skel'])[1])
        E['A3_climber'] = batch(F, skel, G0, pairs,
                                lambda p: inject_single(p, F, skel, +1)[0])
        E['A3_diver'] = batch(F, skel, G0, pairs,
                              lambda p: inject_single(p, F, skel, -1)[0])


        dcage = tc.cage_distance(skel)
        terms = [tuple(free[rng.integers(len(free))]) for _ in range(2 * a.Q)]
        AR = attach_rates(F, skel, terms, dcage)
        print(f'  A3 attachment  climber={AR["climber"]:5.1f}%  '
              f'diver={AR["diver"]:5.1f}%  either={AR["both"]:5.1f}%  '
              f'(steps {AR["climber_steps"]:.1f} / {AR["diver_steps"]:.1f})')



        for k, v in E.items():
            print(f'  {k:16s} feas={v["feas"]:5.1f}%  minclr={v["min_clr"]:.3f} m  '
                  f'{v["ms"]:7.1f} ms  K={v["K"]:.2f}')

        # ---------- A6 parameter sensitivity ----------
        S = dict(tau={}, delta={}, K={}, alpha={})
        print('  -- A6 tau sweep --')
        for tau in [8, 12, 17, 22, 30, 45]:
            c = tc.build_cage(F, tau_deg=tau)
            g_, _ = tc.build_graph(c['skel'], F)
            b = batch(F, c['skel'], g_, pairs[:20],
                      lambda p, sk=c['skel']: tc.inject_gradient(p, F, sk)[1])
            S['tau'][tau] = dict(comps=tc.n_components(c['skel']),
                                 V=g_.number_of_nodes(), E=g_.number_of_edges(),
                                 c_max=c['c_max'], **b)
            print(f'    tau={tau:3d}  comps={S["tau"][tau]["comps"]:2d}  '
                  f'|V|={g_.number_of_nodes():4d}  feas={b["feas"]:5.1f}%  '
                  f'minclr={b["min_clr"]:.3f}')

        print('  -- A6 Delta sweep --')
        for dl in [5, 10, 20, 40, 80]:
            c = tc.build_cage(F, delta=dl)
            g_, _ = tc.build_graph(c['skel'], F)
            b = batch(F, c['skel'], g_, pairs[:20],
                      lambda p, sk=c['skel']: tc.inject_gradient(p, F, sk)[1])
            S['delta'][dl] = dict(comps=tc.n_components(c['skel']),
                                  V=g_.number_of_nodes(), E=g_.number_of_edges(),
                                  d_cage=c['d_cage'], **b)
            print(f'    Delta={dl:3d}  comps={S["delta"][dl]["comps"]:2d}  '
                  f'|V|={g_.number_of_nodes():4d}  feas={b["feas"]:5.1f}%  '
                  f'minclr={b["min_clr"]:.3f}')

        print('  -- A6 K sweep --')
        for K in [1, 3, 5, 8, 12]:
            b = batch(F, skel, G0, pairs[:20], grad, K=K)
            S['K'][K] = b
            print(f'    K={K:2d}  minclr={b["min_clr"]:.3f}  {b["ms"]:7.1f} ms  '
                  f'Kfound={b["K"]:.2f}')

        print('  -- A6 alpha sweep --')
        for al in [0.25, 0.5, 1.0, 2.0, 4.0]:
            sub = pairs[:20]
            ok, steps, ms = 0, [], []
            for (s, g) in sub:
                t = time.perf_counter()
                _, h, st, good = tc.inject_gradient(s, F, skel, alpha=al)
                ms.append((time.perf_counter() - t) * 1e3)
                ok += good; steps.append(st)
            S['alpha'][al] = dict(success=100.0 * ok / len(sub),
                                  steps=float(np.mean(steps)),
                                  ms=float(np.mean(ms)))
            print(f'    alpha={al:4.2f}  inj.success={100.0*ok/len(sub):5.1f}%  '
                  f'steps={np.mean(steps):6.1f}  {np.mean(ms):.3f} ms')

        R[env] = dict(ablation=E, sweep=S, attach=AR,
                      comp_raw=tc.n_components(raw['skel']),
                      comp_caged=tc.n_components(skel))

    json.dump(R, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')

    print('\n%%%% TABLE tab:ablation (mean over envs) %%%%')
    keys = ['A1_no_closure', 'A3_climber', 'A3_diver', 'full']
    lbl = {'A1_no_closure': 'A1 & no closure', 'A3_climber': 'A3 & climber only',
           'A3_diver': 'A3 & diver only', 'full': '-- & \\textbf{full}'}
    for k in keys:
        f = np.mean([R[e]['ablation'][k]['feas'] for e in R])
        c = np.nanmean([R[e]['ablation'][k]['min_clr'] for e in R])
        t = np.mean([R[e]['ablation'][k]['ms'] for e in R])
        print(f'{lbl[k]} & {f:.0f} & {c:.3f} & {t:.1f} \\\\')


if __name__ == '__main__':
    main()