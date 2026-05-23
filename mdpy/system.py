from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.particle_table import ParticleTable
from mdpy.core.tile_list import TileList
from mdpy.core.pbc import compute_pbc_inv
from mdpy.core.gpu_context import GPUContext


class System:

    def __init__(self, topology, pbc_matrix, cutoff=12.0):
        self.topology = topology
        self.particles = ParticleTable(topology.num_particles)
        self.pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
        self.pbc_inv = compute_pbc_inv(self.pbc_matrix)
        self.cutoff = cutoff

        self.gpu = GPUContext()
        self.gpu.initialize(topology, self.pbc_matrix.flatten())

        self.tile_list = TileList(cutoff, skin=2.0)
        self.force_terms = []

        self._step_count = 0
        self._positions_uploaded = False
        self._velocities_uploaded = False

    def add_force_term(self, term):
        self.force_terms.append(term)
        self.gpu.allocate_energy_accumulator(len(self.force_terms))

    def compute_forces(self):
        self.gpu.zero_forces()
        pbc_2d = self.pbc_matrix.reshape(3, 3)
        self.gpu.set_box_dims(
            abs(float(pbc_2d[0, 0])),
            abs(float(pbc_2d[1, 1])),
            abs(float(pbc_2d[2, 2]))
        )
        for term_index, term in enumerate(self.force_terms):
            self.gpu.zero_energy()
            term.compute(self.gpu, self.tile_list)
            self.gpu.accumulate_energy(term_index)

    def dump_energy(self):
        if self.gpu.d_energy_accumulator is None:
            return {}
        raw = cp.asnumpy(self.gpu.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def step(self, integrator, number_steps=1):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True

        for _ in range(number_steps):
            positions_2d = self.gpu.get_positions_2d()
            if self.tile_list.check_rebuild(positions_2d):
                self.tile_list.rebuild(
                    positions_2d, self.topology,
                    self.pbc_matrix, self.pbc_inv,
                )
            self.compute_forces()
            integrator.step(self.gpu)
            self._step_count += 1

    def minimize(self, minimizer, number_steps=100):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True

        positions_2d = self.gpu.get_positions_2d()
        if self.tile_list.check_rebuild(positions_2d):
            self.tile_list.rebuild(
                positions_2d, self.topology,
                self.pbc_matrix, self.pbc_inv,
            )
        self.compute_forces()
        for _ in range(number_steps):
            minimizer.step(self)
        self.gpu.download_positions(self.particles)

    def dump_state(self):
        self.gpu.download_positions(self.particles)
        self.gpu.download_velocities(self.particles)
        return self.particles.positions.copy(), self.particles.velocities.copy()

    @property
    def step_count(self):
        return self._step_count
