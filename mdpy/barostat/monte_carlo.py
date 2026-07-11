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
    float* __restrict__ prev_pos_z,
    float box_x,
    float box_y,
    float box_z,
    float inv_box_x,
    float inv_box_y,
    float inv_box_z
) {
    int mol = blockIdx.x * blockDim.x + threadIdx.x;
    if (mol >= num_molecules) return;

    int first = molecule_start_index[mol];
    int last = molecule_start_index[mol + 1];

    // Use first atom as reference for PBC unwrapping.
    int ref_atom = molecule_atoms[first];
    float rx = pos_x[ref_atom];
    float ry = pos_y[ref_atom];
    float rz = pos_z[ref_atom];

    // Compute centroid using minimum-image unwrapping: for each atom,
    // compute the shortest displacement from the reference atom (unwrapping
    // across PBC boundaries), then reconstruct the unwrapped position.
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int i = first; i < last; i++) {
        int atom = molecule_atoms[i];
        float dx = pos_x[atom] - rx;
        float dy = pos_y[atom] - ry;
        float dz = pos_z[atom] - rz;
        // Minimum image: shift by integer multiples of box to get shortest distance
        dx -= box_x * floorf(dx * inv_box_x + 0.5f);
        dy -= box_y * floorf(dy * inv_box_y + 0.5f);
        dz -= box_z * floorf(dz * inv_box_z + 0.5f);
        // Accumulate unwrapped position (ref + unwrapped displacement)
        cx += rx + dx;
        cy += ry + dy;
        cz += rz + dz;
    }
    float inv_n = 1.0f / (float)(last - first);
    cx *= inv_n; cy *= inv_n; cz *= inv_n;

    // Scale about the centroid (same delta for all atoms preserves internal geometry)
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

    Requires ``state.set_particle_molecule_ids()`` to have been called before
    the first ``apply()``.

    Args:
        pressure_bar: Target external pressure in bar.
        temperature: Simulation temperature in Kelvin.
        frequency: Attempt a volume move every `frequency` steps (default 25).
    """

    name = 'monte_carlo_barostat'

    def __init__(self, pressure_bar, temperature, frequency=25):
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
        self._molecule_csr_built = False

        self._scale_kernel = None

    def _build_molecule_csr(self, state):
        mol_ids = cp.asnumpy(state.d_particle_molecule_ids)
        molecule_atoms, molecule_start_index = build_molecule_csr(mol_ids)
        self._d_molecule_atoms = cp.asarray(molecule_atoms)
        self._d_molecule_start_index = cp.asarray(molecule_start_index)
        self._num_molecules = len(molecule_start_index) - 1
        self._molecule_csr_built = True

    def _ensure_scale_kernel(self):
        if self._scale_kernel is None:
            self._scale_kernel = cp.RawKernel(_SCALE_POSITIONS_KERNEL, 'scale_molecule_positions_kernel')

    def _scale_positions(self, state, scale):
        """Scale positions about each molecule's centroid by `scale`.

        Uses reference-atom + minimum-image unwrapping to correctly compute
        centroids for molecules that straddle PBC boundaries.

        Translates both d_positions and d_prev_positions by the same delta
        so velocity = (pos - prev_pos) / dt is preserved.
        """
        if not self._molecule_csr_built:
            raise RuntimeError("Molecule CSR not built. Call apply(system) first.")
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
                np.float32(state.box_x),
                np.float32(state.box_y),
                np.float32(state.box_z),
                np.float32(state.inv_box_x),
                np.float32(state.inv_box_y),
                np.float32(state.inv_box_z),
            ),
        )

    def apply(self, system):
        """Attempt a Monte Carlo volume move (called every step; acts every `frequency`)."""
        self._step += 1
        if self._step < self.frequency:
            return
        self._step = 0

        state = system.state

        if not self._molecule_csr_built:
            self._build_molecule_csr(state)

        volume = state.box_x * state.box_y * state.box_z

        if self._volume_scale is None:
            self._volume_scale = 0.01 * volume

        energy_initial = system.compute_total_energy()

        delta_volume = self._volume_scale * (2.0 * np.random.random() - 1.0)
        new_volume = volume + delta_volume
        if new_volume <= 0:
            return
        scale_factor = np.float32((new_volume / volume) ** (1.0 / 3.0))

        saved_pos_x = state.d_positions_x.copy()
        saved_pos_y = state.d_positions_y.copy()
        saved_pos_z = state.d_positions_z.copy()
        saved_prev_x = state.d_prev_positions_x.copy()
        saved_prev_y = state.d_prev_positions_y.copy()
        saved_prev_z = state.d_prev_positions_z.copy()
        saved_pbc = state.d_pbc_matrix.get().reshape(3, 3).copy()

        self._scale_positions(state, scale_factor)

        new_pbc = saved_pbc * float(scale_factor)
        system.resize_box(new_pbc)

        energy_final = system.compute_total_energy()

        delta_energy = energy_final - energy_initial
        kT = BOLTZMANN * self.temperature
        log_v_ratio = np.log(new_volume / volume)
        weight = (delta_energy
                  + self.pressure * delta_volume
                  - self._num_molecules * kT * log_v_ratio)

        accept = weight <= 0.0 or np.random.random() < np.exp(-weight / kT)

        self._num_attempted += 1

        if accept:
            self._num_accepted += 1
        else:
            state.d_positions_x[:] = saved_pos_x
            state.d_positions_y[:] = saved_pos_y
            state.d_positions_z[:] = saved_pos_z
            state.d_prev_positions_x[:] = saved_prev_x
            state.d_prev_positions_y[:] = saved_prev_y
            state.d_prev_positions_z[:] = saved_prev_z
            system.resize_box(saved_pbc)

        if self._num_attempted >= 10:
            rate = self._num_accepted / self._num_attempted
            if rate < 0.25:
                self._volume_scale /= 1.1
            elif rate > 0.75:
                self._volume_scale = min(self._volume_scale * 1.1, 0.3 * volume)
            self._num_attempted = 0
            self._num_accepted = 0

    @property
    def acceptance_rate(self):
        if self._num_attempted == 0:
            return 0.0
        return self._num_accepted / self._num_attempted
