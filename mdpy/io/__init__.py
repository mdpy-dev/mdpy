__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"


# Parameter file
from .charmm_toppar_parser import CharmmTopparParser

# Topology file
from .psf_parser import PSFParser

# Position file
from .pdb_parser import PDBParser
from .pdb_writer import PDBWriter

# Comprehensive file

# HDF5
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
from .hdf5_parser import HDF5Parser
from .hdf5_writer import HDF5Writer

__all__ = [ 
    'CharmmTopparParser',
    'PSFParser',
    'PDBParser','PDBWriter',
    'HDF5Parser', 'HDF5Writer',
]