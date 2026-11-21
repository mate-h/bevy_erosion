#!/usr/bin/env python3
"""Check the headless file runner (examples/headless.rs) end to end. Standard library only.

1. A synthetic 300 x 200 heightfield at 10 m (not a power of two, so the pyramid pads it) with a
   softness mask and an outflow strip: every output exists at the input size and is finite, the
   terrain eroded, the outflow strip kept its heights, and two runs differ by dz RMS < 1 cm.
2. Experiment 6's inputs, if present (~/datasets/terracraft/erosion_upres_test/inputs, or
   $EROSION_EXP6_INPUTS): the 2048 x 2048 bicubic upres at 1 m with the WorldCover erodibility as
   softness, experiment 6's settings. Outputs go to the work directory, never into the dataset.

The repro check (two runs, dz RMS < 1 cm) is reported for each run; --strict makes a failing one
fail the script.

usage: scripts/headless_check.py [WORK_DIR] [--skip-exp6] [--strict] [--quiet]
       (WORK_DIR default target/headless_check)
"""
import array, json, math, os, subprocess, sys

QUIET = '--quiet' in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'target', 'release', 'examples', 'headless')
OUTPUTS = ['eroded_height', 'deposition', 'erosion', 'flow', 'drainage_area', 'age',
           'momentum_x', 'momentum_y', 'rivers']
RUN_KEYS = ['width', 'height', 'dx', 'base_m', 'metres_per_height', 'levels', 'frames',
            'cascade_seconds', 'total_seconds', 'seed', 'git', 'adapter', 'inputs', 'params',
            'outputs', 'repro']


def write_f32(path, values):
    a = array.array('f', values)
    if sys.byteorder != 'little':
        a.byteswap()
    with open(path, 'wb') as f:
        a.tofile(f)


def read_f32(path):
    a = array.array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    if sys.byteorder != 'little':
        a.byteswap()
    return a


REPRO_FAILURES = []


def run(args):
    print('$', ' '.join(args), flush=True)
    r = subprocess.run(args, cwd=ROOT, stderr=subprocess.DEVNULL if QUIET else None)
    # 3: the outputs are written but two runs differed by more than the repro limit.
    if r.returncode not in (0, 3):
        raise SystemExit(f'runner exited with {r.returncode}')


def check_outputs(out, w, h):
    run_json = json.load(open(os.path.join(out, 'run.json')))
    missing = [k for k in RUN_KEYS if k not in run_json]
    assert not missing, f'run.json lacks {missing}'
    assert (run_json['width'], run_json['height']) == (w, h), 'run.json size'
    fields = {}
    for name in OUTPUTS:
        v = read_f32(os.path.join(out, name + '.f32'))
        assert len(v) == w * h, f'{name}: {len(v)} texels, expected {w * h}'
        assert all(math.isfinite(x) for x in v), f'{name} holds NaN or infinity'
        fields[name] = v
    repro = run_json['repro']
    assert repro and 'dz_rms_m' in repro, 'run.json has no repro check'
    lp = ', '.join(f"{k} {v * 1000:.1f}" for k, v in repro['dz_rms_lowpass_m'].items())
    print(f"  ok: {w}x{h}, cascade {run_json['cascade_seconds']:.2f} s, levels {run_json['levels']}, "
          f"adapter {run_json['adapter']['name']}")
    verdict = 'pass' if repro['pass'] else 'FAIL'
    print(f"  repro {verdict}: dz rms {repro['dz_rms_m'] * 1000:.1f} mm (limit "
          f"{repro['limit_rms_m'] * 1000:.0f} mm, max {repro['dz_max_m'] * 1000:.0f} mm; "
          f"low-passed mm: {lp})")
    if not repro['pass']:
        REPRO_FAILURES.append(out)
    return run_json, fields


def synthetic(work):
    w, h, dx = 300, 200, 10.0
    d = os.path.join(work, 'synthetic')
    os.makedirs(os.path.join(d, 'inputs'), exist_ok=True)
    heights, soft, outflow = [], [], []
    for y in range(h):
        for x in range(w):
            # A ridge running across the map, valleys on its flanks, and some uneven detail.
            u, v = x / w, y / h
            # Slopes up to about 25 degrees, as on real hillsides.
            z = 400 + 300 * math.exp(-((v - 0.5) / 0.3) ** 2) + 40 * math.sin(9 * u + 2 * v)
            z += 8 * math.sin(41 * u) * math.cos(37 * v) + 2 * math.sin(97 * u + 71 * v)
            heights.append(z)
            soft.append(0.2 + 0.8 * u)
            outflow.append(1.0 if x < 12 else 0.0)  # a draining strip down the west edge
    write_f32(os.path.join(d, 'inputs', 'height.f32'), heights)
    write_f32(os.path.join(d, 'inputs', 'softness.f32'), soft)
    write_f32(os.path.join(d, 'inputs', 'outflow.f32'), outflow)
    json.dump({'width': w, 'height': h, 'metres_per_texel': dx, 'origin': [0.0, 0.0]},
              open(os.path.join(d, 'inputs', 'height.json'), 'w'))
    json.dump({'feature_size': 16, 'erosion_amount': 0.1, 'seed': 7},
              open(os.path.join(d, 'params.json'), 'w'))
    out = os.path.join(d, 'sim_raw')
    run([EXE, '--height', f'{d}/inputs/height.f32', '--softness', f'{d}/inputs/softness.f32',
         '--outflow', f'{d}/inputs/outflow.f32', '--params', f'{d}/params.json', '--out', out])
    run_json, f = check_outputs(out, w, h)
    dz = [a - b for a, b in zip(f['eroded_height'], heights)]
    moved = math.sqrt(sum(z * z for z in dz) / len(dz))
    assert moved > 0.05, f'terrain barely changed (dz rms {moved:.3f} m)'
    strip = [abs(dz[y * w + x]) for y in range(h) for x in range(2, 10)]
    rest = [abs(dz[y * w + x]) for y in range(h) for x in range(40, w)]
    strip_mean, rest_mean = sum(strip) / len(strip), sum(rest) / len(rest)
    assert strip_mean < 0.1 * rest_mean, \
        f'outflow strip moved {strip_mean:.3f} m against {rest_mean:.3f} m elsewhere'
    assert run_json['params']['mask_source'] == 1 and run_json['params']['use_outflow'] == 1
    assert run_json['params']['trail_seed'] == 7
    assert max(f['drainage_area']) > 100 * dx * dx, 'drainage area did not accumulate'
    print(f'  ok: dz rms {moved:.2f} m; outflow strip |dz| {strip_mean:.4f} m vs {rest_mean:.3f} m')


def exp6(work):
    src = os.environ.get('EROSION_EXP6_INPUTS',
                         os.path.expanduser('~/datasets/terracraft/erosion_upres_test/inputs'))
    if not os.path.exists(os.path.join(src, 'upres_bicubic.f32')):
        print(f'  skipped: no experiment-6 inputs at {src}')
        return
    d = os.path.join(work, 'exp6')
    os.makedirs(d, exist_ok=True)
    # Experiment 6's run (sim_raw/run.json there): feature size 16, erosion amount 0.1, a bedrock
    # 4 m down, softness from WorldCover, no rivers.
    json.dump({'feature_size': 16, 'erosion_amount': 0.1, 'use_bedrock': 1, 'bedrock_depth_m': 4,
               'seed': 1}, open(os.path.join(d, 'params.json'), 'w'))
    out = os.path.join(d, 'sim_raw')
    run([EXE, '--height', f'{src}/upres_bicubic.f32', '--grid', f'{src}/grid.json',
         '--softness', f'{src}/erodibility.f32', '--params', f'{d}/params.json', '--out', out])
    grid = json.load(open(os.path.join(src, 'grid.json')))
    _, f = check_outputs(out, grid['width'], grid['height'])
    up = read_f32(os.path.join(src, 'upres_bicubic.f32'))
    dz = [a - b for a, b in zip(f['eroded_height'], up)]
    print(f'  dz vs bicubic: rms {math.sqrt(sum(z * z for z in dz) / len(dz)):.3f} m, '
          f'mean {sum(dz) / len(dz):.3f} m')
    old = os.path.join(os.path.dirname(src), 'sim_raw', 'eroded_height.f32')
    if os.path.exists(old):
        o = read_f32(old)
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(f['eroded_height'], o)) / len(o))
        print(f'  against experiment 8\'s old sim_raw (older engine, not expected equal): '
              f'rms {diff:.3f} m')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    work = os.path.abspath(args[0] if args else os.path.join(ROOT, 'target', 'headless_check'))
    subprocess.run(['cargo', 'build', '--release', '--example', 'headless'], cwd=ROOT, check=True)
    print('synthetic heightfield')
    synthetic(work)
    if '--skip-exp6' not in sys.argv:
        print('experiment 6 inputs')
        exp6(work)
    if REPRO_FAILURES:
        print(f'repro check failed for {len(REPRO_FAILURES)} run(s): {REPRO_FAILURES}')
        if '--strict' in sys.argv:
            raise SystemExit(1)
    print('headless check passed' + (' (repro check failed, see above)' if REPRO_FAILURES else ''))


if __name__ == '__main__':
    main()
