
"""
exp2_connectivity.py -> Table "Effect of closure on connectivity and query
                        feasibility"  +  injection-mechanism comparison (A2)

Answers: does the super-critical level set actually merge the disconnected
skeleton, and does gradient injection beat the alternatives?

Usage:  python exp2_connectivity.py [--Q 100] [--T 200]
"""
import argparse, json, math, time
import numpy as np
import networkx as nx
import tc_core as tc


def feasibility(F, skel, G0, pairs, inject='gradient'):
    ok = 0
    for (s, g) in pairs:
        if inject == 'gradient':
            _, hs, _, a_ = tc.inject_gradient(s, F, skel)
            _, hg, _, b_ = tc.inject_gradient(g, F, skel)
        else:
            hs, _, _, a_ = tc.inject_astar_to_skel(s, F, skel)
            hg, _, _, b_ = tc.inject_astar_to_skel(g, F, skel)
        if not (a_ and b_):
            continue
        G = G0.copy()
        ns = tc.attach(G, F, skel, hs, 's')
        ng = tc.attach(G, F, skel, hg, 'g')
        if ns is None or ng is None:
            continue
        try:
            nx.shortest_path(G, ns, ng, weight='weight'); ok += 1
        except Exception:
            pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=int, default=100)
    ap.add_argument('--T', type=int, default=200, help='randomised terminals for injection test')
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    ap.add_argument('--out', default='results_exp2.json')
    a = ap.parse_args()

    R = {}
    for env in a.envs.split(','):
        print(f'\n===== {env} =====', flush=True)
        mask = tc.make_env(env)
        F = tc.Fields(mask)
        rng = np.random.default_rng(7)
        free = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)

        pairs = []
        while len(pairs) < a.Q:
            s = tuple(free[rng.integers(len(free))])
            g = tuple(free[rng.integers(len(free))])
            if math.dist(s, g) > 120:
                pairs.append((s, g))

        caged = tc.build_cage(F, use_closure=True)
        raw = tc.build_cage(F, use_closure=False)
        Gc, _ = tc.build_graph(caged['skel'], F)
        Gr, _ = tc.build_graph(raw['skel'], F)

        c_raw = tc.n_components(raw['skel'])
        c_cag = tc.n_components(caged['skel'])
        f_raw = feasibility(F, raw['skel'], Gr, pairs)
        f_cag = feasibility(F, caged['skel'], Gc, pairs)
        print(f'  components  raw={c_raw}  caged={c_cag}')
        print(f'  feasibility raw={100*f_raw/a.Q:.1f}%  caged={100*f_cag/a.Q:.1f}%')

        # ---------- A2: injection mechanisms ----------
        terms = [tuple(free[rng.integers(len(free))]) for _ in range(a.T)]
        skel = caged['skel']
        inj = {}

        ok, ms, steps = 0, [], []
        for p in terms:
            t = time.perf_counter()
            _, hit, st, good = tc.inject_gradient(p, F, skel)
            ms.append((time.perf_counter() - t) * 1e3)
            ok += good; steps.append(st)
        inj['gradient'] = dict(success=100 * ok / a.T, ms=float(np.mean(ms)),
                               steps=float(np.mean(steps)), collisions=0.0)

        ok, ms, coll = 0, [], 0
        for p in terms:
            hit, t_, good, c = tc.inject_nearest_pixel(p, F, skel)
            ms.append(t_); ok += good; coll += c
        inj['nearest_pixel'] = dict(success=100 * ok / a.T,
                                    ms=float(np.mean(ms[-a.T:])),
                                    steps=float('nan'),
                                    collisions=100 * coll / a.T)

        ok, ms, exps = 0, [], []
        for p in terms:
            hit, t_, e, good = tc.inject_astar_to_skel(p, F, skel)
            ms.append(t_); ok += good; exps.append(e)
        inj['astar'] = dict(success=100 * ok / a.T, ms=float(np.mean(ms[-a.T:])),
                            steps=float(np.mean(exps)), collisions=0.0)

        ok, ms = 0, []
        for p in terms:
            hit, t_, good = tc.inject_raycast(p, F, Gc)
            ms.append(t_); ok += good
        inj['raycast'] = dict(success=100 * ok / a.T, ms=float(np.mean(ms[-a.T:])),
                              steps=float('nan'), collisions=0.0)

        for k, v in inj.items():
            print(f'  inject {k:14s} success={v["success"]:5.1f}%  '
                  f'{v["ms"]:7.3f} ms  steps/exp={v["steps"]:.1f}  '
                  f'collide={v["collisions"]:.1f}%')

        R[env] = dict(comp_raw=c_raw, comp_caged=c_cag,
                      feas_raw=100 * f_raw / a.Q, feas_caged=100 * f_cag / a.Q,
                      Q=a.Q, T=a.T, injection=inj,
                      c_max=caged['c_max'], d_cage=caged['d_cage'])

    json.dump(R, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')

    print('\n%%%% TABLE tab:connectivity %%%%')
    for env, d in R.items():
        print(f'{env} & {d["comp_raw"]} & {d["comp_caged"]} & '
              f'{d["feas_raw"]:.0f}\\% & {d["feas_caged"]:.0f}\\% \\\\')

    print('\n%%%% A2 injection ablation rows %%%%')
    for env, d in R.items():
        print(f'% {env}')
        for k, lbl in [('nearest_pixel', 'A2a & nearest-pixel'),
                       ('astar', 'A2b & A$^\\star$ inject'),
                       ('raycast', 'A2c & raycast inject'),
                       ('gradient', '-- & \\textbf{gradient (ours)}')]:
            v = d['injection'][k]
            print(f'{lbl} & {v["success"]:.0f} & {v["ms"]:.3f} & '
                  f'{v["collisions"]:.1f} \\\\')


if __name__ == '__main__':
    main()