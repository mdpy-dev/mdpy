from __future__ import annotations

from mdpy.force.force_term import ForceTerm


class ForceGroup(ForceTerm):
    name = ''

    def __init__(self, forces):
        if not forces:
            raise ValueError("ForceGroup requires at least one force")
        force_type = type(forces[0])
        for f in forces:
            if type(f) is not force_type:
                raise TypeError(
                    f"ForceGroup requires homogeneous types, "
                    f"got {force_type.__name__} and {type(f).__name__}"
                )
        self._forces = list(forces)
        self._force_type = force_type
        self._merged_nonbonded = None
        self._do_compile()

    def _do_compile(self):
        from mdpy.force.nonbonded_force import NonbondedForce
        if all(isinstance(f, NonbondedForce) for f in self._forces):
            self._compile_nonbonded()

    def _compile_nonbonded(self):
        from mdpy.force.nonbonded_force import NonbondedForce

        merged_expr = self._forces[0]._expression
        for f in self._forces[1:]:
            merged_expr = merged_expr + f._expression

        cutoffs = [f._cutoff for f in self._forces if f._cutoff is not None]
        cutoff = cutoffs[0] if cutoffs else 12.0

        self._merged_nonbonded = NonbondedForce(merged_expr, cutoff)

        for f in self._forces:
            for name, mat in f._pair_param_data.items():
                if name not in self._merged_nonbonded._pair_param_data:
                    self._merged_nonbonded.set_pair_parameter(name, mat)
            for name, val in f._scalar_data.items():
                if name not in self._merged_nonbonded._scalar_data:
                    self._merged_nonbonded.set_scalar(name, val)

    def __add__(self, other):
        if isinstance(other, ForceGroup):
            if other._force_type is not self._force_type:
                raise TypeError(
                    f"Cannot add ForceGroup of {other._force_type.__name__} "
                    f"to ForceGroup of {self._force_type.__name__}"
                )
            group = ForceGroup(self._forces + other._forces)
            group.name = self.name
            return group
        if type(other) is self._force_type:
            group = ForceGroup(self._forces + [other])
            group.name = self.name
            return group
        if isinstance(other, ForceTerm) and type(other) is not self._force_type:
            raise TypeError(
                f"Cannot add {type(other).__name__} to ForceGroup of "
                f"{self._force_type.__name__}"
            )
        return NotImplemented

    def compute(self, state, block_list=None, compute_energy=True):
        if self._merged_nonbonded is not None:
            self._merged_nonbonded.compute(state, block_list, compute_energy)
        else:
            for f in self._forces:
                f.compute(state, block_list, compute_energy)

    def remap_indices_gpu(self, d_remap):
        if self._merged_nonbonded is not None:
            return
        for f in self._forces:
            f.remap_indices_gpu(d_remap)

    @property
    def sub_forces(self):
        return self._forces

    def bind_sorted(self, topology, block_list, state):
        if self._merged_nonbonded is not None:
            self._merged_nonbonded.bind_sorted(topology, block_list, state)
        else:
            for f in self._forces:
                if hasattr(f, 'bind_sorted'):
                    f.bind_sorted(topology, block_list, state)
