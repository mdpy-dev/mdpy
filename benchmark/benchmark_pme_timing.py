"""PME sub-kernel timing and correctness reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_pme_timing.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cupy as cp
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.system import System

DATA_1M9Z = os.path.join(os.path.dirname(__file__), "..", "mdpy", "test", "data")
DATA_ION = os.path.join(os.path.dirname(__file__), "data")


def setup_system(data_dir, psf_name, pdb_name, prm_names, box):
    psf = PSFParser(os.path.join(data_dir, psf_name))
    pdb = PDBParser(os.path.join(data_dir, pdb_name))
    toppar = CharmmTopparParser(*[os.path.join(data_dir, p) for p in prm_names])
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.diag(box)
    forces = create_charmm_forces(topology, pt, pbc, cutoff=12.0)
    system = System(topology)
    system.set_pbc(pbc)
    system.add_force_term(forces["bonded"])
    system.add_force_term(forces["nonbonded"])
    system.add_force_term(forces["pme"])
    n = topology.num_particles
    system.set_positions(pdb.positions)
    system.set_velocities(np.zeros((n, 3), dtype=np.float32))
    system.update_neighbor_list(force_rebuild=True)
    system.compute_forces()
    return system, forces


def time_pme(system, forces, n_iter=500):
    """Time the PME compute() call in isolation."""
    pme = forces["pme"]
    state = system.state
    bl = system._block_list

    # Warmup
    for _ in range(20):
        state.d_forces_x[:] = 0; state.d_forces_y[:] = 0; state.d_forces_z[:] = 0
        if state.d_energy is not None:
            state.d_energy[:] = 0
        pme.compute(state, bl)

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        state.d_forces_x[:] = 0; state.d_forces_y[:] = 0; state.d_forces_z[:] = 0
        if state.d_energy is not None:
            state.d_energy[:] = 0
        pme.compute(state, bl)
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def main():
    print("=" * 70)
    print("  PME Timing Benchmark")
    print("=" * 70)

    # Ion system
    system_ion, forces_ion = setup_system(
        DATA_ION, "ion.psf", "ion.pdb",
        ["par_sin.prm", "par_water.prm"],
        np.array([75.450, 77.623, 69.668]),
    )
    t_ion = time_pme(system_ion, forces_ion)
    pme_ion = forces_ion["pme"]
    print(f"\n  Ion ({system_ion.topology.num_particles} atoms):")
    print(f"    Grid: {pme_ion.grid_x}x{pme_ion.grid_y}x{pme_ion.grid_z}")
    print(f"    Alpha: {pme_ion.alpha:.4f}")
    print(f"    PME time: {t_ion:.4f} ms")

    # 1M9Z system
    system_1m9z, forces_1m9z = setup_system(
        DATA_1M9Z, "1M9Z.psf", "1M9Z.pdb",
        ["par_all36_prot.prm", "toppar_water_ions.str"],
        np.array([108.0, 108.0, 108.0]),
    )
    t_1m9z = time_pme(system_1m9z, forces_1m9z)
    pme_1m9z = forces_1m9z["pme"]
    print(f"\n  1M9Z ({system_1m9z.topology.num_particles} atoms):")
    print(f"    Grid: {pme_1m9z.grid_x}x{pme_1m9z.grid_y}x{pme_1m9z.grid_z}")
    print(f"    Alpha: {pme_1m9z.alpha:.4f}")
    print(f"    PME time: {t_1m9z:.4f} ms")

    # Save reference energy + forces for correctness checking
    energies_ion = system_ion.dump_energy()
    forces_ion_arr = system_ion.dump_forces()
    energies_1m9z = system_1m9z.dump_energy()
    forces_1m9z_arr = system_1m9z.dump_forces()

    np.savez(
        os.path.join(os.path.dirname(__file__), "pme_reference.npz"),
        ion_energy=energies_ion.get("pme_reciprocal", 0.0),
        ion_forces=forces_ion_arr,
        ion_energy_all=list(energies_ion.items()),
        m19z_energy=energies_1m9z.get("pme_reciprocal", 0.0),
        m19z_forces=forces_1m9z_arr,
        m19z_energy_all=list(energies_1m9z.items()),
    )
    print(f"\n  Reference saved to benchmark/pme_reference.npz")
    print(f"  Ion  PME energy: {energies_ion.get('pme_reciprocal', 0.0):.6e}")
    print(f"  1M9Z PME energy: {energies_1m9z.get('pme_reciprocal', 0.0):.6e}")


if __name__ == "__main__":
    main()
