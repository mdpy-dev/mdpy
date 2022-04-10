__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"

from .constraint import Constraint
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

from .electrostatic_cutoff_constraint import ElectrostaticCutoffConstraint
from .electrostatic_pme_constraint import ElectrostaticPMEConstraint

# Charmm
from .charmm_bond_constraint import CharmmBondConstraint
from .charmm_angle_constraint import CharmmAngleConstraint
from .charmm_dihedral_constraint import CharmmDihedralConstraint
from .charmm_improper_constraint import CharmmImproperConstraint
from .charmm_vdw_constraint import CharmmVDWConstraint

__all__ = [
    'check_long_range_solver',
    'ElectrostaticCutoffConstraint',
    'ElectrostaticPMEConstraint',
    'CharmmBondConstraint',
    'CharmmAngleConstraint',
    'CharmmDihedralConstraint',
    'CharmmImproperConstraint',
    'CharmmVDWConstraint'
]