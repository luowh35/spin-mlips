#!/usr/bin/env python3
"""Integration regression: python test_mpi.py --lammps /path/to/lmp.

Requires MPI, numpy and torch. Uses temporary files, a deterministic nonlinear
3-hop spin model, and an independent full periodic graph/autograd reference.
No GPU is required. --device auto exercises automatic GPU placement when present.
--model optionally tests an existing single-element Fe STEP TorchScript export.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict

import numpy as np
import torch


class SpinModel(torch.nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        pos, m = data['pos'], data['magmoms']
        dst, src = data['edge_index'][0], data['edge_index'][1]
        r = pos[src] - pos[dst] + data['shifts']
        w = (1.0 - (r * r).sum(1) / 2.4**2).clamp(min=0.0)**3
        h = (m * m).sum(1) * 0.05 + m[:, 0] * 0.03
        e = h * 0.1 + 0.2  # nonzero on-site energy for isolated atoms
        for _ in range(3):
            a = torch.zeros_like(h).index_add(0, dst, w * h[src])
            h = torch.tanh(h + 0.2 * a)
            e = e + h * 0.1
        return e.unsqueeze(1)


def load_cpu(path):
    extra = {'config.json': ''}
    model = torch.jit.load(str(path), map_location='cpu', _extra_files=extra).eval()
    def visit(block):
        for node in block.nodes():
            if node.kind() == 'prim::Constant' and list(node.outputs()) and str(node.output().type()) == 'Device':
                node.s_('value', 'cpu')
            for sub in node.blocks():
                visit(sub)
    for module in model.modules():
        for name in module._c._method_names():
            visit(module._c._get_method(name).graph)
    return model, json.loads(extra['config.json'])


def oracle(model, xyz, moments, cell, periodic, cutoff):
    pos = torch.tensor(xyz, dtype=torch.float32, requires_grad=True)
    mag = torch.tensor(moments, dtype=torch.float32, requires_grad=True)
    # Independent O(N^2) full-cell graph with explicit periodic shifts.
    pairs, shifts = [], []
    ranges = [range(-2, 3) if p else [0] for p in periodic]
    for shift in itertools.product(*ranges):
        delta = np.array(shift) @ cell
        for i in range(len(xyz)):
            for j in range(len(xyz)):
                if i == j and not any(shift):
                    continue
                if np.linalg.norm(xyz[j] - xyz[i] + delta) < cutoff:
                    pairs.append((i, j))
                    shifts.append(delta)
    edge = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2).T.contiguous()
    sh = torch.tensor(np.array(shifts).reshape(-1, 3), dtype=torch.float32)
    eps = torch.zeros(3, 3, requires_grad=True)
    deform = torch.eye(3) + eps
    e = model({'pos': pos @ deform, 'magmoms': mag,
               'numbers': torch.zeros(len(xyz), dtype=torch.int64),
               'edge_index': edge, 'shifts': sh @ deform})
    grads = torch.autograd.grad(e.sum(), (pos, mag, eps), allow_unused=True)
    g = [torch.zeros_like(x) if v is None else v for x, v in zip((pos, mag, eps), grads)]
    return e.detach().numpy().reshape(-1), -g[0].numpy(), -g[1].numpy(), -g[2].numpy()


def write_data(path, xyz, moments, cell):
    lines = ['STEP MPI regression', '', f'{len(xyz)} atoms', '1 atom types', '',
             f'0 {cell[0,0]} xlo xhi', f'0 {cell[1,1]} ylo yhi', f'0 {cell[2,2]} zlo zhi']
    if np.any(np.tril(cell, -1)):
        lines.append(f'{cell[1,0]} {cell[2,0]} {cell[2,1]} xy xz yz')
    lines += ['', 'Masses', '', '1 55.845', '', 'Atoms # spin', '']
    for i, (x, m) in enumerate(zip(xyz, moments), 1):
        mag = np.linalg.norm(m)
        direction = m / mag
        # atom_style spin data columns: x y z spx spy spz magnitude.
        lines.append(' '.join(map(str, [i, 1, *x, *direction, mag])))
    path.write_text('\n'.join(lines) + '\n')


def read_dump(path):
    lines = path.read_text().splitlines()
    start = max(i for i, line in enumerate(lines) if line.startswith('ITEM: ATOMS')) + 1
    return np.array([[float(x) for x in line.split()] for line in lines[start:]])


def run_case(root, exe, mpiexec, ranks, batch, device, model, xyz, moments, cell,
             periodic, steps=0, extra='', processors=None, integrator='nve/spin/sib lattice moving'):
    directory = root / f'run{len(list(root.glob("run*")))}'
    directory.mkdir()
    write_data(directory / 'atoms.data', xyz, moments, cell)
    text = f'''units metal
atom_style spin
atom_modify map array
boundary {' '.join('p' if p else 'f' for p in periodic)}
newton on
'''
    if processors:
        text += f'processors {processors}\n'
    text += f'''read_data atoms.data
pair_style spin/step device {device} batch_size {batch}
pair_coeff * * {model} Fe
neighbor 0.4 bin
neigh_modify delay 0 every 1 check yes
compute field all property/atom fmx fmy fmz sp spx spy spz
compute ep all pe/atom
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
thermo_modify format float %.12g
thermo 1
timestep 0.000001
'''
    if steps:
        text += f'fix integrate all {integrator}\n'
    text += extra + '\n'
    text += '''dump result all custom 1 result.dump id x y z fx fy fz c_field[1] c_field[2] c_field[3] c_ep c_field[4] c_field[5] c_field[6] c_field[7]
dump_modify result sort id format float %.12g
'''
    text += f'run {steps}\n'
    (directory / 'in.step').write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cmd = [mpiexec, '-np', str(ranks), str(exe), '-in', 'in.step']
    result = subprocess.run(cmd, cwd=directory, env=env, capture_output=True, text=True, timeout=180)
    (directory / 'stdout.txt').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{cmd} failed ({directory}):\n{result.stdout[-5000:]}\n{result.stderr[-2000:]}')
    thermo = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 8:
            try:
                values = [float(v) for v in fields]
                if values[0] == steps:
                    thermo = values
            except ValueError:
                pass
    if not thermo:
        raise RuntimeError(f'No thermo output in {directory}')
    return read_dump(directory / 'result.dump'), np.array(thermo[1:])


def sib_half(model, x, m, cell, cutoff, dt):
    """Reference transverse SIB step, solving 3x3 linear systems directly."""
    magnitude = np.linalg.norm(m, axis=1, keepdims=True)
    initial = m / magnitude
    hbar = 4.135667403e-3 / (2 * np.pi)

    def solve(field):
        result = []
        for spin, (fx, fy, fz) in zip(initial, field * dt):
            cross = np.array([[0., -fz, fy], [fz, 0., -fx], [-fy, fx, 0.]])
            new = np.linalg.solve(np.eye(3) - 0.5 * cross, (np.eye(3) + 0.5 * cross) @ spin)
            result.append(new / np.linalg.norm(new))
        return np.array(result)

    field = oracle(model, x, m, cell, [False]*3, cutoff)[2] * magnitude / hbar
    midpoint = 0.5 * (initial + solve(field))
    field_mid = oracle(model, x, midpoint * magnitude, cell, [False]*3, cutoff)[2] * magnitude / hbar
    return solve(field_mid) * magnitude


def sib_regressions(root, exe, args, modelpath, model, xyz, moments, cell):
    # Adding an integrator must not double forces at run 0.
    for ranks in [1, 2]:
        plain, thermo = run_case(root, exe, args.mpiexec, ranks, 5, args.device,
                                  modelpath, xyz, moments, cell, [True]*3)
        for integrator in ['nve/spin/sib lattice moving',
                           'nvt/spin/sib temp 1 1 1',
                           'npt/spin/sib temp 1 1 1 iso 0 0 1']:
            with_fix, wt = run_case(root, exe, args.mpiexec, ranks, 5, args.device,
                                    modelpath, xyz, moments, cell, [True]*3,
                                    extra=f'fix integrate all {integrator}')
            np.testing.assert_allclose(with_fix[:,4:11], plain[:,4:11], atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(wt, thermo, atol=1e-6, rtol=1e-6)
            print(f'PASS setup forces: {integrator}, ranks={ranks}', flush=True)

    # All three SIB integrators must execute both spin half-steps. Freeze the
    # lattice and make the NPT barostat effectively stationary for comparison.
    frozen, _ = run_case(root, exe, args.mpiexec, 1, 5, args.device,
                          modelpath, xyz, moments, cell, [True]*3, steps=2,
                          integrator='nve/spin/sib lattice frozen', extra='timestep 0.001')
    for ranks in [1, 2]:
        for style in ['nvt/spin/sib temp 1 1 1 lattice frozen',
                      'npt/spin/sib temp 1 1 1 iso 0 0 1e10 lattice frozen']:
            data, _ = run_case(root, exe, args.mpiexec, ranks, 5, args.device,
                               modelpath, xyz, moments, cell, [True]*3, steps=2,
                               integrator=style, extra='timestep 0.001')
            np.testing.assert_allclose(data[:,11:15], frozen[:,11:15], atol=2e-6, rtol=2e-6)
            print(f'PASS two spin half-steps: {style}, ranks={ranks}', flush=True)
    sib_migration_regression(root, exe, args, modelpath, model, moments)


def sib_migration_regression(root, exe, args, modelpath, model, moments):
    # The pair moves from outside cutoff+skin to inside cutoff in one step,
    # and atom 1 migrates to another rank. Its second spin half-step MUST see
    # the rebuilt neighbor list. Zero initial pair force makes the drift exact.
    box = np.diag([12., 8., 7.])
    x = np.array([[4.7, 4., 3.5], [7.3, 4., 3.5]])
    m = moments[:2]
    velocity = np.array([[150.,0,0], [50.,0,0]])
    dt = 0.01
    mid_m = sib_half(model, x, m, box, 2.4, dt/2)
    new_x = x + dt * velocity
    expected_m = sib_half(model, new_x, mid_m, box, 2.4, dt/2)
    for ranks in [1, 2, 4]:
        data, _ = run_case(root, exe, args.mpiexec, ranks, 1, args.device,
                           modelpath, x, m, box, [False]*3, steps=1,
                           processors=f'{ranks} 1 1', extra="""neighbor 0.01 bin
neigh_modify delay 0 every 1 check yes
group a id 1
group b id 2
velocity a set 150 0 0 units box
velocity b set 50 0 0 units box
timestep 0.01""")
        np.testing.assert_allclose(data[:,1:4], new_x, atol=1e-6)
        np.testing.assert_allclose(data[:,11:12] * data[:,12:15], expected_m, atol=2e-6, rtol=2e-6)
        print(f'PASS new neighbors + migration: ranks={ranks}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lammps', required=True, type=Path)
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--baseline', type=Path, help='Optional old executable: require SIB regressions to fail')
    parser.add_argument('--workdir', type=Path, help='Keep generated files here')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.workdir or Path(tempfile.mkdtemp(prefix='step-mpi-test-'))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='suite-', dir=root))
    exe = args.lammps.resolve()
    fixture = root / 'model.pt'
    config = {'r_max': 2.4, 'num_layers': 3, 'atom_types_map': {'26': 0},
              'project_target_mag_force': False}
    torch.jit.save(torch.jit.script(SpinModel()), str(fixture),
                   _extra_files={'config.json': json.dumps(config)})
    modelpath = args.model.resolve() if args.model else fixture.resolve()
    model, config = load_cpu(modelpath)
    cutoff = config['r_max']
    cell = np.diag([8., 7., 6.])
    xyz = np.array(list(itertools.product([0.3, 2.1, 4.0, 6.1], [0.4, 2.5, 4.8], [0.5, 2.6, 4.5])))
    rng = np.random.default_rng(129)
    xyz += rng.uniform(-0.08, 0.08, xyz.shape)
    moments = rng.normal(size=xyz.shape)
    moments *= 2.2 / np.linalg.norm(moments, axis=1, keepdims=True)
    hbar = 6.582119569e-4  # eV ps; LAMMPS constants differ by ~1e-8 relative
    max_error = 0.
    cases = [('periodic', xyz, moments, cell, [True]*3),
             ('triclinic', xyz, moments, cell + np.array([[0,0,0],[0.6,0,0],[0.3,0.2,0]]), [True]*3)]
    if not args.model:
        cases += [('empty-ranks', xyz[:3], moments[:3], np.diag([30., 8., 7.]), [False]*3),
                  ('isolated', np.array([[1., 1., 1.]]), moments[:1], np.diag([30., 8., 7.]), [False]*3)]
    for name, x, m, box, pbc in cases:
        if name == 'triclinic':
            x = x @ np.linalg.inv(cell) @ box
        expected_e, expected_f, expected_m, expected_v = oracle(model, x, m, box, pbc, cutoff)
        if config.get('project_target_mag_force', False):
            unit = m / np.linalg.norm(m, axis=1, keepdims=True)
            expected_m -= np.sum(expected_m * unit, axis=1, keepdims=True) * unit
        expected_field = expected_m * np.linalg.norm(m, axis=1, keepdims=True) / hbar
        expected_thermo = np.r_[expected_e.sum(), expected_v[[0,1,2,0,0,1],[0,1,2,1,2,2]] / np.linalg.det(box) * 1602176.634]
        for ranks, batch in [(1, 0), (1, 5), (2, 0), (2, 5), (4, 5)]:
            data, thermo = run_case(root, exe, args.mpiexec, ranks, batch, args.device,
                                   modelpath, x, m, box, pbc,
                                   processors=f'{ranks} 1 1' if name in ('empty-ranks', 'isolated') else None)
            np.testing.assert_allclose(data[:, 11:12] * data[:, 12:15], m, atol=1e-10)
            np.testing.assert_allclose(data[:, 4:7], expected_f, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(data[:, 7:10] * hbar, expected_field * hbar, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(data[:, 10], expected_e, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(thermo, expected_thermo, atol=0.3, rtol=3e-4)
            max_error = max(max_error, np.max(np.abs(data[:,4:7] - expected_f)))
            print(f'PASS {name}: ranks={ranks}, batch={batch}', flush=True)
    # Deterministic SIB trajectory tests the collective magnetic-only path,
    # communication of changed spin direction/magnitude and cached fields.
    for extra in ['', 'fix longitudinal all glangevin/spin/sib 0 0.1 1.0 12345']:
        base, bt = run_case(root, exe, args.mpiexec, 1, 0, args.device, modelpath,
                           xyz, moments, cell, [True]*3, steps=3, extra=extra)
        for ranks in [2, 4]:
            data, thermo = run_case(root, exe, args.mpiexec, ranks, 5, args.device,
                                   modelpath, xyz, moments, cell, [True]*3, steps=3, extra=extra)
            np.testing.assert_allclose(data[:,1:7], base[:,1:7], atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(data[:,7:10]*hbar, base[:,7:10]*hbar, atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(data[:,10:], base[:,10:], atol=3e-4, rtol=3e-4)
            print(f'PASS SIB {"longitudinal" if extra else "transverse"}: ranks={ranks}', flush=True)
    if not args.model:
        sib_regressions(root, exe, args, modelpath, model, xyz, moments, cell)
        if args.baseline:
            for label, check in [
                    ('setup', lambda: sib_regressions(root, args.baseline.resolve(), args, modelpath, model, xyz, moments, cell)),
                    ('new neighbors', lambda: sib_migration_regression(root, args.baseline.resolve(), args, modelpath, model, moments))]:
                try:
                    check()
                except AssertionError as error:
                    print(f'CONFIRMED old SIB fails {label}: {str(error)[:200]}', flush=True)
                else:
                    raise AssertionError(f'Old SIB unexpectedly passed {label}')
    print(f'All checks passed; max atomic force error={max_error:.3g}. Files: {root}')


if __name__ == '__main__':
    main()
