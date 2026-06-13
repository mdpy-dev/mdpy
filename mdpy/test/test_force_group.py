import pytest
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.force_group import ForceGroup
from mdpy.force.force_term import ForceTerm


@bonded_expression(body=2)
def dummy_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr


@bonded_expression(body=2)
def dummy_bond_b(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    return k * r


def test_force_term_add_returns_group():
    f1 = BondedForce(dummy_bond)
    f2 = BondedForce(dummy_bond_b)
    group = f1 + f2
    assert isinstance(group, ForceGroup)
    assert len(group._forces) == 2


def test_group_add_force_term():
    f1 = BondedForce(dummy_bond)
    f2 = BondedForce(dummy_bond_b)
    group = f1 + f2
    f3 = BondedForce(dummy_bond)
    group2 = group + f3
    assert isinstance(group2, ForceGroup)
    assert len(group2._forces) == 3


def test_group_add_group():
    f1 = BondedForce(dummy_bond)
    f2 = BondedForce(dummy_bond_b)
    f3 = BondedForce(dummy_bond)
    f4 = BondedForce(dummy_bond_b)
    g1 = f1 + f2
    g2 = f3 + f4
    g3 = g1 + g2
    assert isinstance(g3, ForceGroup)
    assert len(g3._forces) == 4


def test_mixed_type_add_raises():
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    f1 = BondedForce(dummy_bond)
    f2 = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
    with pytest.raises(TypeError):
        f1 + f2


def test_empty_group_raises():
    with pytest.raises(ValueError):
        ForceGroup([])
