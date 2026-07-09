"""Write mdpy simulation state to a PDB file.

Column positions match PDBParser._parse() exactly, ensuring round-trip
fidelity: a file written by PDBWriter can be read back by PDBParser.
"""

import numpy as np
from mdpy.io.pdb_parser import _guess_element


class PDBWriter:
    """Write mdpy simulation state to a PDB file.

    Atom metadata (names, residue info, chain info) is provided at
    construction time in PDB order, aligning with positions/velocities
    from system.dump_state().

    Parameters
    ----------
    particle_ids : array-like of int, shape (N,)
        Atom serial numbers.
    particle_names : list of str, length N
        Atom names ("CA", "N", "CB", ...).
    particle_molecule_ids : array-like of int, shape (N,)
        Residue sequence numbers.
    particle_molecule_types : list of str, length N
        Residue names ("VAL", "PHE", "HOH", ...).
    particle_chain_ids : list of str, length N
        Chain identifiers ("A", "B", ...), each exactly 1 character.
    """

    def __init__(
        self,
        particle_ids,
        particle_names,
        particle_molecule_ids,
        particle_molecule_types,
        particle_chain_ids,
    ):
        self._particle_ids = np.asarray(particle_ids, dtype=np.int32)
        self._particle_names = list(particle_names)
        self._particle_molecule_ids = np.asarray(particle_molecule_ids, dtype=np.int32)
        self._particle_molecule_types = list(particle_molecule_types)
        self._particle_chain_ids = list(particle_chain_ids)
        self._num_particles = len(self._particle_ids)

    def write(
        self,
        file_path,
        positions,
        velocities=None,
        pbc_matrix=None,
    ):
        """Write a single-frame PDB file.

        Parameters
        ----------
        file_path : str
            Output file path (should end with .pdb).
        positions : ndarray of shape (N, 3)
            Atomic positions in Angstrom, PDB order.
        velocities : ndarray of shape (N, 3), optional
            If provided, written as B-factor column (velocity magnitude).
        pbc_matrix : ndarray of shape (3, 3), optional
            Periodic box matrix. Writes CRYST1 record using diagonal
            elements and 90 degree angles (orthorhombic assumption).
        """
        pos = np.asarray(positions, dtype=np.float64)
        n = pos.shape[0]

        lines = []

        # --- CRYST1 record ---
        if pbc_matrix is not None:
            pbc = np.asarray(pbc_matrix, dtype=np.float64)
            a = float(pbc[0, 0])
            b = float(pbc[1, 1])
            c = float(pbc[2, 2])
            lines.append(
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                f"  90.00  90.00  90.00 P 1           1\n"
            )

        # --- ATOM records ---
        for i in range(n):
            x = pos[i, 0]
            y = pos[i, 1]
            z = pos[i, 2]

            serial = self._particle_ids[i]
            atom_name = self._particle_names[i]
            # Truncate residue name to 3 chars (PDB standard)
            residue_name = self._particle_molecule_types[i][:3]
            chain = self._particle_chain_ids[i]
            resid = self._particle_molecule_ids[i]

            # Temperature factor: velocity magnitude if provided
            if velocities is not None:
                vx = float(velocities[i, 0])
                vy = float(velocities[i, 1])
                vz = float(velocities[i, 2])
                vmag = np.sqrt(vx * vx + vy * vy + vz * vz)
                beta = min(float(vmag), 99.99)
            else:
                beta = 0.00

            # Element symbol from atom name
            elem = _guess_element(atom_name)

            # PDB column format (matching PDBParser._parse):
            #   1-6:   "ATOM  "
            #   7-11:  serial   (5d)
            #   12:    " "      (space)
            #   13-16: name     (4s, left)
            #   17:    " "      (alt loc)
            #   18-20: res_name (3s, right)
            #   21:    " "      (space)
            #   22:    chain    (1c)
            #   23-26: res_id   (4d)
            #   27-30: "    "   (spaces)
            #   31-38: x        (8.3f)
            #   39-46: y        (8.3f)
            #   47-54: z        (8.3f)
            #   55-60: "  1.00" (occupancy)
            #   61-66: beta     (6.2f)
            #   67-76: " " * 10 (spaces)
            #   77-78: element  (2s, right)
            # Build using exact column positions for ATOM records.
            # Columns use 1-based indexing (matching PDB spec).
            # Parser slices: [6:11]=serial, [12:16]=name, [17:21]=residue,
            #                [21]=chain, [22:26]=resid, [30:38]=x,
            #                [38:46]=y, [46:54]=z.
            line = (
                f"ATOM  "                               #  1-6
                f"{serial:5d}"                          #  7-11
                f" "                                    # 12
                f"{atom_name:<4.4s}"                    # 13-16
                f" "                                    # 17  (alt loc)
                f"{residue_name:>3.3s}"                 # 18-20
                f" "                                    # 21
                f"{chain}"                              # 22
                f"{resid:4d}"                           # 23-26
                f"    "                                 # 27-30
                f"{x:8.3f}"                             # 31-38
                f"{y:8.3f}"                             # 39-46
                f"{z:8.3f}"                             # 47-54
                f"  1.00"                               # 55-60
                f"{beta:6.2f}"                          # 61-66
                f"          "                           # 67-76
                f"{elem:>2s}"                           # 77-78
                f"\n"
            )
            lines.append(line)

        lines.append("END\n")

        with open(file_path, "w") as f:
            f.writelines(lines)
