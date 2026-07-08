from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.core.block_list import BlockList
from mdpy.core.state import State


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

        self._step_counter = 0

    def _ensure_pme_stream(self):
        if self._pme_stream is None:
            self._pme_stream = cp.cuda.Stream(non_blocking=True)
            # disable_timing: these are hot-path per-step ordering events;
            # timing-enabled events add ~1-2 us of sync overhead each.
            self._ev_zero_forces = cp.cuda.Event(disable_timing=True)
            self._ev_pme_done = cp.cuda.Event(disable_timing=True)

    def set_pbc(self, pbc_matrix):
        self.state.set_pbc(pbc_matrix)

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
                term.compute(self.state, self._block_list, compute_energy=False)
            return

        # PME runs on a non-blocking stream concurrent with primary terms.
        # zero_forces just ran on the null stream; record an event so the PME
        # stream does not atomicAdd into d_forces before the zeroing completes.
        self._ev_zero_forces.record()

        # Launch PME pipeline on its stream, gated on zero_forces.
        self._pme_stream.wait_event(self._ev_zero_forces)
        with self._pme_stream:
            for term in self._pme_force_terms:
                term.compute(self.state, self._block_list, compute_energy=False)
            # Record INSIDE the with-block so the event is recorded on the
            # pme stream, not the null stream (record() uses the current stream).
            self._ev_pme_done.record()

        # Primary terms run on the null stream, overlapping with PME.
        for term in self._primary_force_terms:
            term.compute(self.state, self._block_list, compute_energy=False)

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

