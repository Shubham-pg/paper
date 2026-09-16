
"""
exp1_multiquery.py  ->  Table "Cost over 100 consecutive queries"
                        Table "Per-query breakdown and graph size"

Runs Q randomised start-goal pairs per environment through every planner and
reports offline cost, average query cost, and the stage-wise split for ours.

Usage:  python exp1_multiquery.py [--Q 100] [--envs E1,E2,E3,E4]
"""
import argparse, json, time, math
import numpy as np
import tc_core as tc


def sample_pairs(F, Q, seed=0):
    rng = np.random.default_rng(seed)
    free = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)
    pairs = []
    while len(pairs) < Q:
        a = tuple(free[rng.integers(len(free))])
        b = tuple(free[rng.integers(len(free))])
        if math.dist(a, b) > 120:
            pairs.append((a, b))
    return pairs


def run_ours(F, cage, G0, pairs):
    skel = cage['skel']
    rec = dict(inject=[], enum=[], select=[], tension=[], total=[], ok=0,
               steps=[], minclr=[], length=[])
    for (s, g) in pairs:
        t_all = time.perf_counter()
        t = time.perf_counter()
        _, hs, ss, oks = tc.inject_gradient(s, F, skel)
        _, hg, sg, okg = tc.inject_gradient(g, F, skel)
        t_inj = (time.perf_counter() - t) * 1e3
        if not (oks and okg):
            rec['inject'].append(t_inj); continue
        rec['steps'] += [ss, sg]

        G = G0.copy()
        ns = tc.attach(G, F, skel, hs, 's')
        ng = tc.attach(G, F, skel, hg, 'g')

        t = time.perf_counter()
        npaths = tc.enumerate_paths(G, ns, ng)
        t_enum = (time.perf_counter() - t) * 1e3
        if not npaths:
            rec['inject'].append(t_inj); rec['enum'].append(t_enum); continue

        cands = [{'pixels': tc.node_path_to_pixels(G, p)} for p in npaths]
        t = time.perf_counter()
        best, _ = tc.select_path(cands, F, 'lex')
        t_sel = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        tr, ts, _ = tc.selective_relaxation(best['pixels'], F)
        band = tc.tension_band(best['pixels'], F, tr, ts)
        t_ten = (time.perf_counter() - t) * 1e3

        rec['inject'].append(t_inj); rec['enum'].append(t_enum)
        rec['select'].append(t_sel); rec['tension'].append(t_ten)
        rec['total'].append((time.perf_counter() - t_all) * 1e3)
        rec['ok'] += 1
        rec['minclr'].append(best['min_clr']); rec['length'].append(best['len'])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=int, default=100)
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    ap.add_argument('--skip-astar', action='store_true')
    ap.add_argument('--out', default='results_exp1.json')
    a = ap.parse_args()

    R = {}
    for env in a.envs.split(','):
        print(f'\n===== {env} =====', flush=True)
        mask = tc.make_env(env)
        F = tc.Fields(mask)
        pairs = sample_pairs(F, a.Q, seed=hash(env) % 1000)

        # ---------- OURS: offline ----------
        t = time.perf_counter()
        F2 = tc.Fields(mask)                       # field build is part of offline
        cage = tc.build_cage(F2)
        G0, t_graph = tc.build_graph(cage['skel'], F2)
        off_ours = (time.perf_counter() - t) * 1e3
        print(f'  ours offline {off_ours:7.1f} ms  |V|={G0.number_of_nodes()} '
              f'|E|={G0.number_of_edges()}  C_max={cage["c_max"]:.1f} '
              f'd_cage={cage["d_cage"]:.1f}', flush=True)

        r = run_ours(F2, cage, G0, pairs)
        print(f'  ours query  {np.mean(r["total"]):7.1f} ms   feasible '
              f'{r["ok"]}/{a.Q}', flush=True)

        # ---------- A* ----------
        astar = dict(t=[], exp=[], ok=0)
        if not a.skip_astar:
            for i, (s, g) in enumerate(pairs):
                t = time.perf_counter()
                p, e, ok = tc.astar_grid(F, s, g)
                astar['t'].append((time.perf_counter() - t) * 1e3)
                astar['exp'].append(e); astar['ok'] += ok
                if i % 20 == 0:
                    print(f'    A* {i}/{a.Q}', flush=True)
            print(f'  A*   query  {np.mean(astar["t"]):7.1f} ms  '
                  f'exp={np.mean(astar["exp"]):.0f}', flush=True)

        # ---------- PRM ----------
        t = time.perf_counter()
        Gp, pts = tc.build_prm(F, n_samples=2500, k=12)
        off_prm = (time.perf_counter() - t) * 1e3
        prm = dict(t=[], ok=0)
        for (s, g) in pairs:
            t = time.perf_counter()
            p, ok = tc.query_prm(Gp, pts, F, s, g)
            prm['t'].append((time.perf_counter() - t) * 1e3); prm['ok'] += ok
        print(f'  PRM  offline {off_prm:7.1f} ms  query {np.mean(prm["t"]):7.1f} ms  '
              f'feasible {prm["ok"]}/{a.Q}', flush=True)

        # ---------- RRT* ----------
        rrt = dict(t=[], ok=0)
        for i, (s, g) in enumerate(pairs):
            t = time.perf_counter()
            p, it, ok = tc.rrt_star(F, s, g, seed=i)
            rrt['t'].append((time.perf_counter() - t) * 1e3); rrt['ok'] += ok
        print(f'  RRT* query  {np.mean(rrt["t"]):7.1f} ms  feasible '
              f'{rrt["ok"]}/{a.Q}', flush=True)

        # ---------- GVD + A* injection ----------
        raw = tc.build_cage(F, use_closure=False)
        rawskel = raw['skel']
        Graw, _ = tc.build_graph(rawskel, F)
        gvd = dict(t=[], ok=0, exp=[])
        for (s, g) in pairs:
            t = time.perf_counter()
            hs, _, es, oks = tc.inject_astar_to_skel(s, F, rawskel)
            hg, _, eg, okg = tc.inject_astar_to_skel(g, F, rawskel)
            if oks and okg:
                G = Graw.copy()
                ns = tc.attach(G, F, rawskel, hs, 's')
                ng = tc.attach(G, F, rawskel, hg, 'g')
                try:
                    import networkx as nx
                    nx.shortest_path(G, ns, ng, weight='weight'); gvd['ok'] += 1
                except Exception:
                    pass
                gvd['exp'] += [es, eg]
            gvd['t'].append((time.perf_counter() - t) * 1e3)
        print(f'  GVD+A* query {np.mean(gvd["t"]):7.1f} ms  feasible '
              f'{gvd["ok"]}/{a.Q}  inj.expansions={np.mean(gvd["exp"] or [0]):.0f}',
              flush=True)

        # ---------- EGVG-style raycast injection on the caged graph ----------
        eg_ = dict(t=[], ok=0)
        for (s, g) in pairs:
            t = time.perf_counter()
            hs, _, oks = tc.inject_raycast(s, F2, G0)
            hg, _, okg = tc.inject_raycast(g, F2, G0)
            if oks and okg:
                G = G0.copy()
                ns = tc.attach(G, F2, cage['skel'], hs, 's')
                ng = tc.attach(G, F2, cage['skel'], hg, 'g')
                try:
                    import networkx as nx
                    nx.shortest_path(G, ns, ng, weight='weight'); eg_['ok'] += 1
                except Exception:
                    pass
            eg_['t'].append((time.perf_counter() - t) * 1e3)
        print(f'  EGVG-ray query {np.mean(eg_["t"]):6.1f} ms  feasible '
              f'{eg_["ok"]}/{a.Q}', flush=True)

        R[env] = dict(
            Q=a.Q,
            graph=dict(V=G0.number_of_nodes(), E=G0.number_of_edges()),
            cage=dict(c_max=cage['c_max'], d_cage=cage['d_cage']),
            ours=dict(offline_ms=off_ours, query_ms=float(np.mean(r['total'])),
                      query_std=float(np.std(r['total'])), feasible=r['ok'],
                      inject_ms=float(np.mean(r['inject'])),
                      enum_ms=float(np.mean(r['enum'])),
                      select_ms=float(np.mean(r['select'])),
                      tension_ms=float(np.mean(r['tension'])),
                      inject_steps=float(np.mean(r['steps'])),
                      min_clr_m=float(np.mean(r['minclr'])),
                      len_m=float(np.mean(r['length']))),
            astar=dict(offline_ms=0.0,
                       query_ms=float(np.mean(astar['t'])) if astar['t'] else None,
                       expansions=float(np.mean(astar['exp'])) if astar['exp'] else None,
                       feasible=astar['ok']),
            prm=dict(offline_ms=off_prm, query_ms=float(np.mean(prm['t'])),
                     feasible=prm['ok']),
            rrt=dict(offline_ms=0.0, query_ms=float(np.mean(rrt['t'])),
                     feasible=rrt['ok']),
            gvd_astar=dict(offline_ms=raw['t_ms'], query_ms=float(np.mean(gvd['t'])),
                           feasible=gvd['ok'],
                           inj_expansions=float(np.mean(gvd['exp'] or [0]))),
            egvg_ray=dict(offline_ms=off_ours, query_ms=float(np.mean(eg_['t'])),
                          feasible=eg_['ok']),
        )

    json.dump(R, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')

    # ---- LaTeX-ready ----
    print('\n%%%% TABLE tab:compute_benchmarks (per environment) %%%%')
    for env, d in R.items():
        Q = d['Q']
        print(f'% --- {env} ---')
        for key, label in [('astar', 'A$^\\star$'), ('prm', 'PRM (dense)'),
                           ('rrt', 'RRT$^\\star$'),
                           ('gvd_astar', 'GVD + A$^\\star$ inject'),
                           ('egvg_ray', 'EGVG-style~\\cite{qi2025egvg}'),
                           ('ours', '\\textbf{Ours}')]:
            v = d[key]
            if v['query_ms'] is None:
                continue
            tot = (v['offline_ms'] + Q * v['query_ms']) / 1000.0
            print(f'{label} & ${v["offline_ms"]:.1f}$ & ${v["query_ms"]:.1f}$ '
                  f'& ${tot:.2f}$ \\\\')
    print('\n%%%% TABLE tab:breakdown %%%%')
    for env, d in R.items():
        o, g = d['ours'], d['graph']
        print(f'{env} & {o["inject_ms"]:.1f} & {o["enum_ms"]:.1f} & '
              f'{o["select_ms"]:.2f} & {o["tension_ms"]:.1f} & '
              f'{g["V"]}/{g["E"]} \\\\')


if __name__ == '__main__':
    main()


