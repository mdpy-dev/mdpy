from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, NA, default_energy_unit, kelvin
from mdpy.barostat._base import BarostatBase

BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)

BAR_TO_INTERNAL_PRESSURE = float(NA.value) * 1e-32


def build_molecule_csr(particle_molecule_ids):
    """Build CSR format from per-particle molecule IDs.

    Groups atom indices by molecule, producing a flat atom-index array and
    CSR start offsets suitable for GPU kernel consumption.

    Args:
        particle_molecule_ids: array-like of int, one molecule ID per particle
            (PDB order).

    Returns:
        molecule_atoms: np.int32 array of atom indices, grouped by molecule
        molecule_start_index: np.int32 array of size (num_molecules + 1)
    """
    mol_ids = np.asarray(particle_molecule_ids, dtype=np.int32)
    sort_order = np.argsort(mol_ids, kind='stable').astype(np.int32)
    sorted_mol_ids = mol_ids[sort_order]

    boundaries = [0]
    for i in range(1, len(sorted_mol_ids)):
        if sorted_mol_ids[i] != sorted_mol_ids[i - 1]:
            boundaries.append(i)
    boundaries.append(len(sorted_mol_ids))

    return sort_order, np.array(boundaries, dtype=np.int32)


_SCALE_POSITIONS_KERNEL = r"""
extern "C" __global__
void scale_molecule_positions_kernel(
    float scale,
    int num_molecules,
    const int* __restrict__ molecule_atoms,
    const int* __restrict__ molecule_start_index,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ prev_pos_x,
    float* __restrict__ prev_pos_y,
    float* __restrict__ prev_pos_z
) {
    int mol = blockIdx.x * blockDim.x + threadIdx.x;
    if (mol >= num_molecules) return;

    int first = molecule_start_index[mol];
    int last = molecule_start_index[mol + 1];

    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int i = first; i < last; i++) {
        int atom = molecule_atoms[i];
        cx += pos_x[atom];
        cy += pos_y[atom];
        cz += pos_z[atom];
    }
    float inv_n = 1.0f / (float)(last - first);
    cx *= inv_n; cy *= inv_n; cz *= inv_n;

    float dx = cx * (scale - 1.0f);
    float dy = cy * (scale - 1.0f);
    float dz = cz * (scale - 1.0f);

    for (int i = first; i < last; i++) {
        int atom = molecule_atoms[i];
        pos_x[atom] += dx;
        pos_y[atom] += dy;
        pos_z[atom] += dz;
        prev_pos_x[atom] += dx;
        prev_pos_y[atom] += dy;
        prev_pos_z[atom] += dz;
    }
}
"""


class MonteCarloBarostat(BarostatBase):
    """Isotropic Monte Carlo barostat for NPT ensemble.

    Periodically proposes a volume change, scales molecular positions about
    each molecule's centroid, recomputes energy, and accepts/rejects via the
    Metropolis criterion.

    Args:
        pressure_bar: Target external pressure in bar.
        temperature: Simulation temperature in Kelvin.
        frequency: Attempt a volume move every `frequency` steps (default 25).
        particle_molecule_ids: Per-particle molecule IDs (PDB order), e.g.
            from ``psf.particle_molecule_ids``. Used for centroid-based scaling.
    """

    name = 'monte_carlo_barostat'

    def __init__(self, pressure_bar, temperature, frequency=25,
                 particle_molecule_ids=None):
        self.pressure_bar = float(pressure_bar)
        self.pressure = self.pressure_bar * BAR_TO_INTERNAL_PRESSURE
        self.temperature = float(temperature)
        self.frequency = int(frequency)

        self._step = 0
        self._volume_scale = None
        self._num_attempted = 0
        self._num_accepted = 0

        self._d_molecule_atoms = None
        self._d_molecule_start_index = None
        self._num_molecules = 0
        if particle_molecule_ids is not None:
            self._init_molecules(particle_molecule_ids)

        self._scale_kernel = None

    def _init_molecules(self, particle_molecule_ids):
        molecule_atoms, molecule_start_index = build_molecule_csr(particle_molecule_ids)
        self._d_molecule_atoms = cp.asarray(molecule_atoms)
        self._d_molecule_start_index = cp.asarray(molecule_start_index)
        self._num_molecules = len(molecule_start_index) - 1

    def _ensure_scale_kernel(self):
        if self._scale_kernel is None:
            self._scale_kernel = cp.RawKernel(_SCALE_POSITIONS_KERNEL, 'scale_molecule_positions_kernel')

    def _scale_positions(self, state, scale):
        """Scale positions about each molecule's centroid by `scale`.

        Translates both d_positions and d_prev_positions by the same delta
        so velocity = (pos - prev_pos) / dt is preserved.
        """
        if self._num_molecules == 0:
            raise RuntimeError("MonteCarloBarostat requires particle_molecule_ids")
        self._ensure_scale_kernel()
        threads_per_block = 256
        grid = ((self._num_molecules + threads_per_block - 1) // threads_per_block,)
        self._scale_kernel(
            grid, (threads_per_block,),
            (
                np.float32(scale),
                np.int32(self._num_molecules),
                self._d_molecule_atoms,
                self._d_molecule_start_index,
                state.d_positions_x, state.d_positions_y, state.d_positions_z,
                state.d_prev_positions_x, state.d_prev_positions_y, state.d_prev_positions_z,
            ),
        )

    @property
    def acceptance_rate(self):
        if self._num_attempted == 0:
            return 0.0
        return self._num_accepted / self._num_attempted
