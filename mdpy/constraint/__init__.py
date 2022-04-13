__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"

from mdpy.constraint.constraint import Constraint
from mdpy.error import *

# Electrostatic
LONG_RANGE_SOLVER = ['PME', 'CUTOFF']
def check_long_range_solver(solver: str):
    if not solver.upper():
        raise EnsemblePoorDefinedError(
            '%s solver is not supported. ' +
            'Check supported platform with `mdpy.constraint.LONG_RANGE_SOLVER`'
        )
    return solver.upper()

from mdpy.constraint.electrostatic_cutoff_constraint import ElectrostaticCutoffConstraint
from mdpy.constraint.electrostatic_pme_constraint import ElectrostaticPMEConstraint
from mdpy.constraint.electrostatic_fdpe_constraint import ElectrostaticFDPEConstraint

# Charmm
from mdpy.constraint.charmm_bond_constraint import CharmmBondConstraint
from mdpy.constraint.charmm_angle_constraint import CharmmAngleConstraint
from mdpy.constraint.charmm_dihedral_constraint import CharmmDihedralConstraint
from mdpy.constraint.charmm_improper_constraint import CharmmImproperConstraint
from mdpy.constraint.charmm_vdw_constraint import CharmmVDWConstraint

__all__ = [
    'check_long_range_solver',
    'ElectrostaticCutoffConstraint',
    'ElectrostaticPMEConstraint',
    'ElectrostaticFDPEConstraint',
    'CharmmBondConstraint',
    'CharmmAngleConstraint',
    'CharmmDihedralConstraint',
    'CharmmImproperConstraint',
    'CharmmVDWConstraint'
]