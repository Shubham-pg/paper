
"""
exp3_selection.py -> Table "Path quality by selection criterion"
                     + U-trap resolution counts (DWA / RRT* / ours)

Answers P3: does the lexicographic bottleneck order buy clearance over
shortest-length and over plain max-min-clearance, and how often does plain
min-clearance tie?

Usage:  python exp3_selection.py [--Q 100]
"""
import argparse, json, math
import numpy as np
import tc_core as tc


def greedy_dwa(F, s, g, max_steps=1500, lookahead=14, n_head=32):
    """Deliberately myopic DWA-style local planner: samples headings, scores by
    goal progress minus obstacle proximity over a short horizon. It has no
    global term, so a U-trap captures it. Returns (path, escaped)."""
    p = np.array(s, float)
    g = np.array(g, float)
    path = [tuple(p)]
    H, W = F.mask.shape
    for _ in range(max_steps):
        if np.linalg.norm(p - g) < 15:
            return path, True
        best, best_sc = None, -1e18
        for th in np.linspace(0, 2 * np.pi, n_head, endpoint=False):
            d = np.array([math.sin(th), math.cos(th)])
            q = p + d * lookahead
            y, x = int(q[0]), int(q[1])
            if not (0 <= y < H and 0 <= x < W) or F.sdf[y, x] <= tc.PARAMS['rho_safe']:
                continue
            sc = -np.linalg.norm(q - g) + 1.5 * min(F.sdf[y, x], 40.0)
            if sc > best_sc:
                best_sc, best = sc, q
        if best is None:
            return path, False
        if np.linalg.norm(best - p) < 1e-6:
            return path, False
        p = best
        path.append(tuple(p))
        # oscillation detector: revisiting the same cell repeatedly
        if len(path) > 80:
            recent = np.array(path[-80:])
            if recent.std(0).max() < lookahead * 1.2:
                return path, False
    return path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=int, default=100)
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    ap.add_argument('--trap-env', default='E4')
    ap.add_argument('--trap-trials', type=int, default=30)
    ap.add_argument('--out', default='results_exp3.json')
    a = ap.parse_args()

    R = {}
    agg = {c: dict(minclr=[], meanclr=[], length=[], ties=0, n=0)
           for c in ('short', 'minclr', 'lex')}

    for env in a.envs.split(','):
        print(f'\n===== {env} =====', flush=True)
        mask = tc.make_env(env)
        F = tc.Fields(mask)
        cage = tc.build_cage(F)
        skel = cage['skel']
        G0, _ = tc.build_graph(skel, F)
        rng = np.random.default_rng(11)
        free = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)

        per = {c: dict(minclr=[], meanclr=[], length=[], ties=0, n=0)
               for c in ('short', 'minclr', 'lex')}
        nk = []
        done = 0
        guard = 0
        while done < a.Q and guard < a.Q * 12:
            guard += 1
            s = tuple(free[rng.integers(len(free))])
            g = tuple(free[rng.integers(len(free))])
            if math.dist(s, g) < 120:
                continue
            _, hs, _, oks = tc.inject_gradient(s, F, skel)
            _, hg, _, okg = tc.inject_gradient(g, F, skel)
            if not (oks and okg):
                continue
            G = G0.copy()
            ns = tc.attach(G, F, skel, hs, 's')
            ng = tc.attach(G, F, skel, hg, 'g')
            npaths = tc.enumerate_paths(G, ns, ng)
            if len(npaths) < 2:
                continue
            base = [tc.node_path_to_pixels(G, p) for p in npaths]
            nk.append(len(base))
            for crit in ('short', 'minclr', 'lex'):
                cands = [{'pixels': px} for px in base]
                best, ties = tc.select_path(cands, F, crit)
                per[crit]['minclr'].append(best['min_clr'])
                per[crit]['meanclr'].append(best['mean_clr'])
                per[crit]['length'].append(best['len'])
                per[crit]['ties'] += ties
                per[crit]['n'] += 1
            done += 1
        print(f'  usable multi-candidate queries: {done}  mean K={np.mean(nk or [0]):.1f}')
        for crit in ('short', 'minclr', 'lex'):
            p = per[crit]
            if p['n'] == 0:
                continue
            print(f'  {crit:8s} minclr={np.mean(p["minclr"]):.3f} m  '
                  f'meanclr={np.mean(p["meanclr"]):.3f} m  '
                  f'len={np.mean(p["length"]):.2f} m  ties={p["ties"]}')
            for k in ('minclr', 'meanclr', 'length'):
                agg[crit][k] += p[k]
            agg[crit]['ties'] += p['ties']; agg[crit]['n'] += p['n']
        R[env] = {c: dict(min_clr=float(np.mean(per[c]['minclr'])) if per[c]['n'] else None,
                          mean_clr=float(np.mean(per[c]['meanclr'])) if per[c]['n'] else None,
                          length=float(np.mean(per[c]['length'])) if per[c]['n'] else None,
                          ties=per[c]['ties'], n=per[c]['n'])
                  for c in ('short', 'minclr', 'lex')}

    # ---------- U-trap resolution ----------
    print(f'\n===== TRAP: {a.trap_env} =====', flush=True)
    mask = tc.make_env(a.trap_env)
    F = tc.Fields(mask)
    cage = tc.build_cage(F); skel = cage['skel']
    G0, _ = tc.build_graph(skel, F)
    rng = np.random.default_rng(5)
    # start INSIDE the U, goal outside it
    inside = np.argwhere((F.sdf > tc.PARAMS['rho_safe'] + 2))
    inside = inside[(inside[:, 0] > 160) & (inside[:, 0] < 370) &
                    (inside[:, 1] > 260) & (inside[:, 1] < 500)]
    outside = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)
    outside = outside[(outside[:, 1] < 200) | (outside[:, 1] > 600)]

    trap = dict(dwa=0, rrt=0, ours=0, n=a.trap_trials)
    for i in range(a.trap_trials):
        s = tuple(inside[rng.integers(len(inside))])
        g = tuple(outside[rng.integers(len(outside))])
        _, esc = greedy_dwa(F, s, g)
        trap['dwa'] += esc
        _, _, ok = tc.rrt_star(F, s, g, max_iter=4000, seed=i)
        trap['rrt'] += ok
        _, hs, _, o1 = tc.inject_gradient(s, F, skel)
        _, hg, _, o2 = tc.inject_gradient(g, F, skel)
        good = False
        if o1 and o2:
            G = G0.copy()
            ns = tc.attach(G, F, skel, hs, 's'); ng = tc.attach(G, F, skel, hg, 'g')
            paths = tc.enumerate_paths(G, ns, ng)
            if paths:
                cands = [{'pixels': tc.node_path_to_pixels(G, p)} for p in paths]
                best, _ = tc.select_path(cands, F, 'lex')
                tr, ts, nrel = tc.selective_relaxation(best['pixels'], F)
                band = tc.tension_band(best['pixels'], F, tr, ts)
                good = band.get('ok', False) and band['min_clr'] > 0
        trap['ours'] += good
        if i % 10 == 0:
            print(f'  trial {i}: dwa={trap["dwa"]} rrt={trap["rrt"]} ours={trap["ours"]}',
                  flush=True)
    print(f'  TRAP RESULT  DWA {trap["dwa"]}/{trap["n"]}  '
          f'RRT* {trap["rrt"]}/{trap["n"]}  Ours {trap["ours"]}/{trap["n"]}')

    R['_aggregate'] = {c: dict(min_clr=float(np.mean(agg[c]['minclr'])),
                               mean_clr=float(np.mean(agg[c]['meanclr'])),
                               length=float(np.mean(agg[c]['length'])),
                               ties=agg[c]['ties'], n=agg[c]['n'])
                       for c in ('short', 'minclr', 'lex') if agg[c]['n']}
    R['_trap'] = trap
    json.dump(R, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')

    print('\n%%%% TABLE tab:clearance (aggregate over all envs) %%%%')
    lbl = {'short': 'Shortest length', 'minclr': 'Max min-clearance',
           'lex': '\\textbf{Lexicographic}'}
    for c in ('short', 'minclr', 'lex'):
        if c not in R['_aggregate']:
            continue
        d = R['_aggregate'][c]
        print(f'{lbl[c]} & {d["min_clr"]:.3f} & {d["mean_clr"]:.3f} & '
              f'{d["length"]:.2f} & {d["ties"]} \\\\')


if __name__ == '__main__':
    main()






