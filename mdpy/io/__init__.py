__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"


# Parameter file
from mdpy.io.charmm_toppar_parser import CharmmTopparParser

# Topology file
from mdpy.io.psf_parser import PSFParser

# Positions file
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.pdb_writer import PDBWriter
from mdpy.io.dcd_parser import DCDParser
from mdpy.io.xyz_parser import XYZParser
from mdpy.io.xyz_writer import XYZWriter

# Comprehensive file
## HDF5
HDF5_FILE_HIERARCHY = '''Hierarchy of HDF5 file created by mdpy
+-- '/'
|   +--	group "topology"
|   |   +-- group "particles"
|   |   |   +-- dataset "particle_id" int
|   |   |   |
|   |   |   +-- dataset "particle_type" str
|   |   |   |
|   |   |   +-- dataset "particle_name" str
|   |   |   |
|   |   |   +-- dataset "matrix_id" int
|   |   |   |
|   |   |   +-- dataset "molecule_id" int
|   |   |   |
|   |   |   +-- dataset "molecule_type" str
|   |   |   |
|   |   |   +-- dataset "chain_id" str
|   |   |   |
|   |   |   +-- dataset "mass" float
|   |   |   |
|   |   |   +-- dataset "charge" float
|   |   |
|   |   +-- dataset "num_particles" int
|   |   |
|   |   +-- dataset "bonds" int
|   |   |
|   |   +-- dataset "num_bonds" int
|   |   |
|   |   +-- dataset "angles" int
|   |   |
|   |   +-- dataset "num_angles" int
|   |   |
|   |   +-- dataset "dihedrals" int
|   |   |
|   |   +-- dataset "num_dihedrals" int
|   |   |
|   |   +-- dataset "impropers" int
|   |   |
|   |   +-- dataset "num_impropers" int
|   |
|   +-- group "positions"
|   |   +-- dataset "frame-0"
|   |   |
|   |   +-- dataset "frame-1"
|   |   .
|   |   .
|   |   |
|   |   +-- dataset "frame-x"
|   |
|   +-- dataset "pbc_matrix"
|
'''
from mdpy.io.hdf5_parser import HDF5Parser
from mdpy.io.hdf5_writer import HDF5Writer

# Other file
from mdpy.io.log_writer import LogWriter

__all__ = [
    'CharmmTopparParser',
    'PSFParser',
    'PDBParser','PDBWriter',
    'DCDParser',
    'XYZParser', 'XYZWriter',
    'HDF5Parser', 'HDF5Writer',
    'LogWriter'
]