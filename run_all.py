

"""
run_all.py -- run every experiment, merge results, and print one consolidated
LaTeX-ready dump to paste back.

Usage:
    python run_all.py --quick     # ~10 min,  Q=20
    python run_all.py             # full,     Q=100
"""
import argparse, json, os, subprocess, sys, platform
import numpy as np
import tc_core as tc


def sh(cmd):
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--envs', default='E1,E2,E3,E4')
    a = ap.parse_args()
    Q = 20 if a.quick else 100
    Qb = 10 if a.quick else 40

    sh(['exp1_multiquery.py', '--Q', str(Q), '--envs', a.envs])
    sh(['exp2_connectivity.py', '--Q', str(Q), '--T', str(Q * 2), '--envs', a.envs])
    sh(['exp3_selection.py', '--Q', str(Q), '--envs', a.envs,
        '--trap-trials', str(max(10, Q // 3))])
    sh(['exp4_band.py', '--Q', str(Qb), '--envs', a.envs])
    sh(['exp5_ablation.py', '--Q', str(Qb), '--envs', a.envs])

    print('\n\n' + '=' * 72)
    print('CONSOLIDATED OUTPUT -- paste everything below back into the chat')
    print('=' * 72)

    # ---- machine / parameters ----
    print('\n### PLATFORM')
    print('python  :', platform.python_version())
    print('machine :', platform.processor() or platform.machine())
    print('system  :', platform.system(), platform.release())
    try:
        import multiprocessing
        print('cores   :', multiprocessing.cpu_count())
    except Exception:
        pass

    print('\n### PARAMS (Table tab:params)')
    for k, v in tc.PARAMS.items():
        print(f'  {k:16s} = {v}')

    print('\n### ENVIRONMENT GEOMETRY')
    for env in a.envs.split(','):
        m = tc.make_env(env)
        F = tc.Fields(m)
        c = tc.build_cage(F)
        H, W = m.shape
        area_m2 = H * W * tc.PX2M ** 2
        free_pct = 100.0 * F.free.mean()
        print(f'  {env}: {H}x{W} px = {H*tc.PX2M:.1f} x {W*tc.PX2M:.1f} m '
              f'({area_m2:.1f} m^2), free {free_pct:.1f}%, '
              f'max clearance {F.sdf.max()*tc.PX2M:.2f} m, '
              f'C_max={c["c_max"]:.1f} px, d_cage={c["d_cage"]:.1f} px')

    for f in ['results_exp1.json', 'results_exp2.json', 'results_exp3.json',
              'results_exp4.json', 'results_exp5.json']:
        if os.path.exists(f):
            print(f'\n### {f}')
            print(open(f).read())


if __name__ == '__main__':
    main()