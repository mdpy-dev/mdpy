import numpy as np
import pytest

from mdpy.force.expressions.pme_exclusion import (
    pme_exclusion_correction,
    COULOMB_CONST,
)


class TestPmeExclusionExpression:
    def test_body_is_2(self):
        assert pme_exclusion_correction.body == 2

    def test_parameter_names_contains_alpha(self):
        assert 'alpha' in pme_exclusion_correction.parameter_names

    def test_per_particle_has_charge(self):
        per_particle = pme_exclusion_correction.per_particle
        assert 'charge1' in per_particle
        assert 'charge2' in per_particle
        assert per_particle['charge1'] == 'charge'
        assert per_particle['charge2'] == 'charge'

    def test_cuda_fragment_contains_erff(self):
        frag = pme_exclusion_correction.cuda_fragment
        assert 'erff' in frag, f"erff not found in CUDA fragment:\n{frag}"

    def test_cuda_fragment_contains_distance_helper(self):
        frag = pme_exclusion_correction.cuda_fragment
        assert 'pbc_wrap_vec' in frag, "PBC wrap not in fragment"
        assert '_delta_' in frag or 'len_f3' in frag, "distance helper not in fragment"

    def test_cuda_fragment_contains_coulomb_constant(self):
        frag = pme_exclusion_correction.cuda_fragment
        expected = f'{COULOMB_CONST}f'
        assert expected in frag, f"Coulomb constant {expected} not inlined in fragment"

    def test_cuda_fragment_contains_force_projection(self):
        frag = pme_exclusion_correction.cuda_fragment
        assert 'add_force' in frag, "Force projection (add_force) not in fragment"
