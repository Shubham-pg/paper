
"""
exp4_band.py -> Table "Executed trajectory quality"  +  A5 band ablation

Variants: raw skeletal route / full band / no damping / no rest length /
no selective relaxation / end-decimation instead of distributed.

Metrics: integral |kappa| ds, angular jerk (RMS of 3rd heading difference),
node-spacing CV, min clearance, settling iterations, convergence rate.

Usage:  python exp4_band.py [--Q 40]
"""
import argparse, json, math
import numpy as np
import tc_core as tc

VARIANTS = ['raw', 'full', 'no_damping', 'no_rest', 'no_relax', 'end_decim']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Q', type=int, default=40)
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    ap.add_argument('--out', default='results_exp4.json')
    a = ap.parse_args()

    acc = {v: dict(curv=[], jerk=[], cv=[], minclr=[], iters=[], conv=0, n=0,
                   length=[]) for v in VARIANTS}
    per_env = {}

    for env in a.envs.split(','):
        print(f'\n===== {env} =====', flush=True)
        mask = tc.make_env(env)
        F = tc.Fields(mask)
        cage = tc.build_cage(F); skel = cage['skel']
        G0, _ = tc.build_graph(skel, F)
        rng = np.random.default_rng(23)
        free = np.argwhere(F.sdf > tc.PARAMS['rho_safe'] + 2)

        done, guard = 0, 0
        while done < a.Q and guard < a.Q * 15:
            guard += 1
            s = tuple(free[rng.integers(len(free))])
            g = tuple(free[rng.integers(len(free))])
            if math.dist(s, g) < 120:
                continue
            _, hs, _, o1 = tc.inject_gradient(s, F, skel)
            _, hg, _, o2 = tc.inject_gradient(g, F, skel)
            if not (o1 and o2):
                continue
            G = G0.copy()
            ns = tc.attach(G, F, skel, hs, 's'); ng = tc.attach(G, F, skel, hg, 'g')
            paths = tc.enumerate_paths(G, ns, ng)
            if not paths:
                continue
            cands = [{'pixels': tc.node_path_to_pixels(G, p)} for p in paths]
            best, _ = tc.select_path(cands, F, 'lex')
            px = best['pixels']
            if len(px) < 30:
                continue

            tr, ts, _ = tc.selective_relaxation(px, F)
            tr_u = np.full(F.mask.shape, tc.PARAMS['clear_rigid'], np.float32)
            ts_u = np.full(F.mask.shape, tc.PARAMS['clear_soft'], np.float32)

            # raw skeletal route, no optimisation at all
            arr = np.array(px, float)
            acc['raw']['curv'].append(tc.int_abs_curvature(arr))
            acc['raw']['jerk'].append(tc.angular_jerk(arr))
            acc['raw']['cv'].append(tc.spacing_cv(arr))
            acc['raw']['minclr'].append(tc.path_min_clearance(arr, F))
            acc['raw']['length'].append(tc.path_length(arr))
            acc['raw']['iters'].append(0); acc['raw']['n'] += 1

            cfg = {
                'full':       dict(thr=(tr, ts), damping=True,  rest=True,  dec='distributed'),
                'no_damping': dict(thr=(tr, ts), damping=False, rest=True,  dec='distributed'),
                'no_rest':    dict(thr=(tr, ts), damping=True,  rest=False, dec='distributed'),
                'no_relax':   dict(thr=(tr_u, ts_u), damping=True, rest=True, dec='distributed'),
                'end_decim':  dict(thr=(tr, ts), damping=True,  rest=True,  dec='ends'),
            }
            for name, c in cfg.items():
                r = tc.tension_band(px, F, c['thr'][0], c['thr'][1],
                                    damping=c['damping'], rest_length=c['rest'],
                                    decimation=c['dec'],
                                    iter_cap=1200)
                if not r.get('ok'):
                    continue
                d = acc[name]
                d['curv'].append(r['curvature']); d['jerk'].append(r['jerk'])
                d['cv'].append(r['spacing_cv']); d['minclr'].append(r['min_clr'])
                d['iters'].append(r['iters']); d['conv'] += int(r['converged'])
                d['length'].append(r['length']); d['n'] += 1
            done += 1
            if done % 10 == 0:
                print(f'  {done}/{a.Q}', flush=True)
        per_env[env] = done

    def sm(x):
        x = [v for v in x if np.isfinite(v)]
        return (float(np.mean(x)), float(np.std(x))) if x else (float('nan'),) * 2

    out = {}
    print('\n%-12s %-16s %-16s %-14s %-12s %-14s %s' %
          ('variant', 'int|kappa|ds', 'ang.jerk', 'spacing CV',
           'minclr(m)', 'iters', 'conv%'))
    for v in VARIANTS:
        d = acc[v]
        if d['n'] == 0:
            continue
        c, cs = sm(d['curv']); j, js = sm(d['jerk']); cv, cvs = sm(d['cv'])
        mc, mcs = sm(d['minclr']); it, its = sm(d['iters'])
        conv = 100.0 * d['conv'] / d['n'] if v != 'raw' else float('nan')
        out[v] = dict(curv=c, curv_sd=cs, jerk=j, jerk_sd=js, cv=cv, cv_sd=cvs,
                      min_clr=mc, min_clr_sd=mcs, iters=it, iters_sd=its,
                      conv_pct=conv, n=d['n'],
                      length=float(np.mean(d['length'])))
        print('%-12s %7.2f+-%-6.2f %7.4f+-%-6.4f %6.3f+-%-5.3f %6.3f+-%-5.3f '
              '%6.0f+-%-5.0f %5.1f' % (v, c, cs, j, js, cv, cvs, mc, mcs, it, its, conv))

    json.dump(dict(variants=out, per_env=per_env), open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')

    print('\n%%%% TABLE tab:kinematics %%%%')
    lbl = {'raw': 'Raw skeletal', 'no_damping': 'Band, no damping',
           'no_rest': 'Band, no rest length', 'no_relax': 'Band, no relaxation',
           'end_decim': 'Band, end decimation', 'full': '\\textbf{Full}'}
    for v in ['raw', 'no_damping', 'no_rest', 'no_relax', 'end_decim', 'full']:
        if v not in out:
            continue
        d = out[v]
        print(f'{lbl[v]} & ${d["curv"]:.2f}$ & ${d["jerk"]:.4f}$ & '
              f'${d["cv"]:.3f}$ & ${d["min_clr"]:.3f}$ \\\\')


if __name__ == '__main__':
    main()