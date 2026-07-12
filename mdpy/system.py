from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.core.block_list import BlockList
from mdpy.core.state import State
from mdpy.unit import NA
from mdpy.barostat.monte_carlo import build_molecule_csr, SCALE_POSITIONS_KERNEL

BAR_TO_INTERNAL_PRESSURE = float(NA.value) * 1e-32


class System:

    def __init__(self, topology, state=None):
        self.topology = topology
        self.num_particles = topology.num_particles
        if state is None:
            state = State(topology.num_particles)  # caller must set_* before compute
        self.state = state

        self._cutoff = None
        self._skin = 1.0
        self._rebuild_check_interval = 10
        self._block_list = None

        self.force_terms = []
        self._primary_force_terms = []
        self._pme_force_terms = []
        self._pme_stream = None
        self._ev_zero_forces = None
        self._ev_pme_done = None
        self.constraints = []
        self.barostats = []

        self._step_counter = 0

        self._d_molecule_atoms = None
        self._d_molecule_start_index = None
        self._num_molecules = 0
        self._scale_positions_kernel = None

    def _ensure_pme_stream(self):
        if self._pme_stream is None:
            self._pme_stream = cp.cuda.Stream(non_blocking=True)
            # disable_timing: these are hot-path per-step ordering events;
            # timing-enabled events add ~1-2 us of sync overhead each.
            self._ev_zero_forces = cp.cuda.Event(disable_timing=True)
            self._ev_pme_done = cp.cuda.Event(disable_timing=True)

    def _ensure_molecule_csr(self):
        if self._d_molecule_atoms is not None:
            return
        if self.state.d_particle_molecule_ids is None:
            raise RuntimeError(
                "Particle molecule IDs not set. Call state.set_particle_molecule_ids() first."
            )
        mol_ids = cp.asnumpy(self.state.d_particle_molecule_ids)
        molecule_atoms, molecule_start_index = build_molecule_csr(mol_ids)
        self._d_molecule_atoms = cp.asarray(molecule_atoms)
        self._d_molecule_start_index = cp.asarray(molecule_start_index)
        self._num_molecules = len(molecule_start_index) - 1

    def _scale_molecular_positions(self, scale):
        self._ensure_molecule_csr()
        if self._scale_positions_kernel is None:
            self._scale_positions_kernel = cp.RawKernel(
                SCALE_POSITIONS_KERNEL, 'scale_molecule_positions_kernel'
            )
        threads = 256
        grid = ((self._num_molecules + threads - 1) // threads,)
        state = self.state
        self._scale_positions_kernel(
            grid, (threads,),
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

    def _compute_translational_ke(self):
        """Compute molecular COM translational kinetic energy on CPU.

        K_trans = Σ_molecules 0.5 * M_mol * |v_com|^2
        where v_com = Σ(m_i * v_i) / M_mol.
        """
        self._ensure_molecule_csr()
        state = self.state
        velocities_x = state.d_velocities_x.get()
        velocities_y = state.d_velocities_y.get()
        velocities_z = state.d_velocities_z.get()
        masses = state.d_particle_masses.get()
        molecule_atoms = cp.asnumpy(self._d_molecule_atoms)
        molecule_start = cp.asnumpy(self._d_molecule_start_index)
        K_trans = 0.0
        for m in range(self._num_molecules):
            start = molecule_start[m]
            end = molecule_start[m + 1]
            momentum_x = momentum_y = momentum_z = 0.0
            total_mass = 0.0
            for i in range(start, end):
                atom = molecule_atoms[i]
                mass = masses[atom]
                momentum_x += mass * velocities_x[atom]
                momentum_y += mass * velocities_y[atom]
                momentum_z += mass * velocities_z[atom]
                total_mass += mass
            if total_mass > 0:
                K_trans += 0.5 * (momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z) / total_mass
        return K_trans

    def set_pbc(self, pbc_matrix):
        self.state.set_pbc(pbc_matrix)

    def resize_box(self, new_pbc_matrix):
        """Change the simulation box during a run (e.g., barostat).

        Coordinates the full box-change cascade:
        1. State: update PBC matrix + inverse + box dims
        2. Block list: rebuild with new cell grid
        3. Force terms: notify any term with update_box() (e.g., PME)
        """
        self.state.set_pbc(new_pbc_matrix)
        if self._block_list is not None:
            self.update_neighbor_list(force_rebuild=True)
        pbc_2d = np.asarray(new_pbc_matrix).reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        for term in self.force_terms:
            if hasattr(term, 'update_box'):
                term.update_box(box_x, box_y, box_z, self._block_list)

    @property
    def block_list(self):
        if self._block_list is None:
            raise RuntimeError(
                "Neighbor list not initialized. Call update_neighbor_list() first."
            )
        return self._block_list

    @property
    def cutoff(self):
        return self._cutoff

    def add_force_term(self, term, stream=None):
        if stream not in (None, 'pme'):
            raise ValueError(
                f"stream must be None or 'pme', got {stream!r}"
            )
        self.force_terms.append(term)
        if stream == 'pme':
            self._ensure_pme_stream()
            self._pme_force_terms.append(term)
        else:
            self._primary_force_terms.append(term)
        self.state.allocate_energy_accumulator(len(self.force_terms))
        self.state.allocate_virial_accumulator(len(self.force_terms))
        term_cutoff = getattr(term, '_cutoff', None)
        if term_cutoff is not None:
            if self._cutoff is None:
                self._cutoff = term_cutoff
            else:
                self._cutoff = max(self._cutoff, term_cutoff)
            if self._block_list is not None:
                self._block_list.set_cutoff(self._cutoff)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def apply_constraints(self, time_step):
        for constraint in self.constraints:
            constraint.apply(self.state, time_step)

    def add_barostat(self, barostat):
        self.barostats.append(barostat)

    def apply_barostats(self):
        for barostat in self.barostats:
            barostat.apply(self)

    def set_positions(self, positions):
        self.state.set_positions(positions)

    def set_velocities(self, velocities):
        self.state.set_velocities(velocities)

    def _ensure_ready(self):
        if not self.state.is_ready:
            raise RuntimeError(
                "State not fully set: positions/velocities/charges/masses/types/pbc "
                "must all be set before compute.")

    def compute_forces(self):
        self._ensure_ready()
        self.state.zero_forces()
        if self._block_list is not None:
            self._block_list.refresh_sorted_posq(self.state)

        if not self._pme_force_terms:
            for term in self._primary_force_terms:
                term.compute(self.state, self._block_list, compute_energy=False, compute_virial=False)
            return

        # PME runs on a non-blocking stream concurrent with primary terms.
        # zero_forces just ran on the null stream; record an event so the PME
        # stream does not atomicAdd into d_forces before the zeroing completes.
        self._ev_zero_forces.record()

        # Launch PME pipeline on its stream, gated on zero_forces.
        self._pme_stream.wait_event(self._ev_zero_forces)
        with self._pme_stream:
            for term in self._pme_force_terms:
                term.compute(self.state, self._block_list, compute_energy=False, compute_virial=False)
            # Record INSIDE the with-block so the event is recorded on the
            # pme stream, not the null stream (record() uses the current stream).
            self._ev_pme_done.record()

        # Primary terms run on the null stream, overlapping with PME.
        for term in self._primary_force_terms:
            term.compute(self.state, self._block_list, compute_energy=False, compute_virial=False)

        # Make the null stream wait for PME to finish writing forces before
        # compute_forces returns, so the next null-stream op (integrator) sees
        # fully-accumulated forces.
        cp.cuda.Stream.null.wait_event(self._ev_pme_done)

    def update_neighbor_list(self, sync_interval=10, force_rebuild=False):
        if not self.state.has_pbc:
            raise RuntimeError("PBC not set. Call set_pbc() first.")
        self._ensure_ready()

        if self._block_list is None:
            if self._cutoff is None:
                raise RuntimeError(
                    "No cutoff available. Add a force term with cutoff first."
                )
            self._block_list = BlockList(
                self._cutoff, skin=self._skin,
                rebuild_check_interval=self._rebuild_check_interval,
            )
            force_rebuild = True

        if force_rebuild:
            cp.cuda.Stream.null.synchronize()
            self._do_rebuild(force=True)
            self._step_counter = 0
            return

        self._block_list.check_rebuild_async(self.state)
        self._step_counter += 1
        if self._step_counter < sync_interval:
            return

        self._do_rebuild(force=False)
        self._step_counter = 0

    def _do_rebuild(self, *, force=False):
        if not force:
            flag_val = int(self._block_list.d_rebuild_flag[0])
            if flag_val == 0:
                return
        self._block_list.rebuild(self.topology, self.state, force=force)
        self.state.wrap_positions_with_prev_correction()
        self._block_list.capture_snapshot(self.state)
        self._block_list.build_block_pairs(self.topology, self.state)
        self._block_list.refresh_sorted_type_indices(self.state)
        self._block_list.reset_flag()

    def dump_energy(self):
        if self.state.d_energy_accumulator is None:
            return {}
        self.state.zero_forces()
        if self._block_list is not None:
            self._block_list.refresh_sorted_posq(self.state)
        for term_index, term in enumerate(self.force_terms):
            self.state.zero_energy()
            term.compute(self.state, self._block_list, compute_energy=True)
            self.state.set_energy_slot(term_index)
        raw = cp.asnumpy(self.state.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def dump_virial(self):
        if self.state.d_virial_accumulator is None:
            return {}
        self.state.zero_forces()
        if self._block_list is not None:
            self._block_list.refresh_sorted_posq(self.state)
        for term_index, term in enumerate(self.force_terms):
            self.state.zero_energy()
            self.state.zero_virial()
            term.compute(
                self.state, self._block_list,
                compute_energy=False, compute_virial=True,
            )
            self.state.set_virial_slot(term_index)
        raw = cp.asnumpy(self.state.d_virial_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            if np.any(raw[term_index] != 0.0):
                result[term.name] = raw[term_index].reshape(3, 3)
        return result

    def dump_energy_and_virial(self):
        """Compute both energy and virial in one pass (F+E+V variant)."""
        energies = {}
        virials = {}
        if self.state.d_energy_accumulator is None:
            return energies, virials
        self.state.zero_forces()
        if self._block_list is not None:
            self._block_list.refresh_sorted_posq(self.state)
        for term_index, term in enumerate(self.force_terms):
            self.state.zero_energy()
            self.state.zero_virial()
            term.compute(
                self.state, self._block_list,
                compute_energy=True, compute_virial=True,
            )
            self.state.set_energy_slot(term_index)
            self.state.set_virial_slot(term_index)
        raw_e = cp.asnumpy(self.state.d_energy_accumulator)
        raw_v = cp.asnumpy(self.state.d_virial_accumulator)
        for i, term in enumerate(self.force_terms):
            if float(raw_e[i]) != 0.0:
                energies[term.name] = float(raw_e[i])
            if np.any(raw_v[i] != 0.0):
                virials[term.name] = raw_v[i].reshape(3, 3)
        return energies, virials

    def dump_pressure(self, virials=None):
        """Compute instantaneous pressure in bar.

        P = (2K + 2*Tr(d_virial)) / (3V), where K is kinetic energy and
        d_virial is the half-virial (0.5 * sum(r⊗F)).

        Args:
            virials: optional dict of per-term 3x3 virial arrays from a
                prior dump_virial() or dump_energy_and_virial() call.
                If None, recomputes forces with virial (expensive).

        Returns:
            Pressure in bar (float).
        """
        if virials is None:
            virials = self.dump_virial()
        kinetic_energy = self.state.compute_kinetic_energy()
        volume = self.state.box_x * self.state.box_y * self.state.box_z
        virial_trace = sum(float(np.trace(W)) for W in virials.values())
        pressure_internal = (2.0 * kinetic_energy + 2.0 * virial_trace) / (3.0 * volume)
        return pressure_internal / BAR_TO_INTERNAL_PRESSURE

    def compute_current_pressure(self):
        """Compute instantaneous pressure via finite difference (bar).

        Replicates OpenMM's MonteCarloBarostat::computeCurrentPressure:
        scales molecular COMs by (1±delta), computes potential energy at
        each scale, and combines with translational KE:

            P = (2/3) * K_trans / V - (E1 - E2) / dV

        This automatically excludes intra-molecular and constraint
        contributions (COM scaling preserves internal geometry). Does
        not use the virial. Does not require a barostat.

        Returns:
            Pressure in bar (float).
        """
        self._ensure_molecule_csr()
        state = self.state
        V = state.box_x * state.box_y * state.box_z

        saved_pos = np.stack([
            state.d_positions_x.get(),
            state.d_positions_y.get(),
            state.d_positions_z.get(),
        ], axis=1)
        saved_prev = np.stack([
            state.d_prev_positions_x.get(),
            state.d_prev_positions_y.get(),
            state.d_prev_positions_z.get(),
        ], axis=1)
        saved_pbc = state.d_pbc_matrix.get().reshape(3, 3).copy()

        delta = 1e-3
        scale1 = 1.0 + delta
        scale2 = 1.0 - delta

        self._scale_molecular_positions(scale1)
        self.resize_box(saved_pbc * scale1)
        E1 = self.compute_total_energy()

        self._scale_molecular_positions(scale2 / scale1)
        self.resize_box(saved_pbc * scale2)
        E2 = self.compute_total_energy()

        state.d_positions_x[:] = cp.asarray(saved_pos[:, 0])
        state.d_positions_y[:] = cp.asarray(saved_pos[:, 1])
        state.d_positions_z[:] = cp.asarray(saved_pos[:, 2])
        state.d_prev_positions_x[:] = cp.asarray(saved_prev[:, 0])
        state.d_prev_positions_y[:] = cp.asarray(saved_prev[:, 1])
        state.d_prev_positions_z[:] = cp.asarray(saved_prev[:, 2])
        self.resize_box(saved_pbc)

        K_trans = self._compute_translational_ke()
        deltaV = V * (scale1 ** 3 - scale2 ** 3)
        P_internal = (2.0 / 3.0) * K_trans / V - (E1 - E2) / deltaV
        return P_internal / BAR_TO_INTERNAL_PRESSURE

    def compute_total_energy(self):
        """Compute total potential energy on GPU, return as Python float.

        Recomputes all forces + energies from scratch (does not reuse cached
        forces). Runs all terms on the null stream for simplicity. The only
        GPU→CPU transfer is the final scalar read.
        """
        self._ensure_ready()
        if not self.force_terms:
            return 0.0
        self.state.zero_forces()
        if self._block_list is not None:
            self._block_list.refresh_sorted_posq(self.state)
        for term_index, term in enumerate(self.force_terms):
            self.state.zero_energy()
            term.compute(self.state, self._block_list, compute_energy=True)
            self.state.set_energy_slot(term_index)
        cp.cuda.Stream.null.synchronize()
        return float(self.state.d_energy_accumulator.sum())

    def dump_state(self):
        state = self.state
        pos = np.stack([
            state.d_positions_x.get(),
            state.d_positions_y.get(),
            state.d_positions_z.get(),
        ], axis=1)
        vel = np.stack([
            state.d_velocities_x.get(),
            state.d_velocities_y.get(),
            state.d_velocities_z.get(),
        ], axis=1)
        return pos, vel

    def dump_forces(self):
        state = self.state
        return np.stack([
            state.d_forces_x.get(),
            state.d_forces_y.get(),
            state.d_forces_z.get(),
        ], axis=1)

