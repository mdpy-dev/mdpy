# Profile Log

## 00_cpu_origin

- Date: 2021-11-04
- Commit: 0df2654dbedb322e4c9165540cd905fad2e54e86
- Modification: None
- Conclusion:
    - Slow

## 01_cpu_numba_njit

- Date: 2021-11-04
- Commit: 64384b5e3ad9fd4aa10fb1ab7643d20de8f68429
- Modification:
    - Add `kernel` function to each constraint class
    - Add `njit` to each kernel
- Conclusion:
    - 7x faster to 00 version

## 02_cpu_numba_internal_njit

- Date: 2021-11-05
- Commit: 41de5d2cd48f41d82f982c95f9f1d6d90be1c774
- Modification:
    - Replace the kernel function with a `constraint.kernel` method
    - Compile jit(kernel) in the initialization of each constraint
- Conclusion:
    - No big change on computational speed
    - Decreasing time of `import mdpy as md`
    - Supporting user's specification of environment variables

## 03_cpu_numba_electrostatic_io_update

- Date: 2021-11-05
- Commit: 17b5d590db87d4bf1bf5cd7e07c387df92e127ab
- Modification:
    - Update the `bind_ensemble` method of `mdpy.constraint.ElectrostaticConstraint`, removing pair-wise parameter
- Conclusion
    - The IO time of` mdpy.constraint.ElectrostaticConstraint` decreases from 19s to 0.5s

## 04_cpu_numba_cell_list_update

- Date: 2021-11-06
- Commit: b59d679af093be823f03188dd58209c16d3868dd
- Modification:
    - Add `mdpy.core.CellList` to `Ensemble.state` attribute
    - Remove `update_neighbor` method of `mdpy.constraint.CharmmNonbondedConstraint`
    - Add support for `CellList` in `mdpy.constraint.CharmmNonbondedConstraint`
- Conclusion:
    - The IO time of `mdpy.constraint.CharmmNonbondedConstraint` increased dramatically

## 05_cpu_numba_cell_list_v2

- Date: 2021-11-07
- Commit: 13ad6b58b0989cefbf5d4c8f9ea9a7f1ccdbcd4a
- Modification:
    - Add kernel function for `mdpy.core.CellList` class
    - Update kernel function for `mdpy.constraint.CharmmNonbondedConstraint` class
- Conclusion:
    - Shrink IO time for `mdpy.constraint.CharmmNonbondedConstraint` 
    - Current Computation time for kernel function `mdpy.constraint.CharmmNonbondedConstraint` is comparable to **03 version** in small system. Good

## 06_cpu_numba_topology_electrostatic_constraint_update

- Date: 2021-11-07
- Commit: 9ac62a0c8aa556c59caef6381d5ea08c97089d5f
- Modification:
    - Add `join` and `split` method to mdpy.core.Topology class
    - Add `bonded_particle` params to the kernel function of `mdpy.constraint.ElectrostaticConstraint` class
    - Update `bind_ensemble` method of `mdpy.constraint.CharmmNonbondedConstraint`, removing most of IO to `mdpy.core.Topology.join()`
- Conclusion
    - Fix bug of `mdpy.constraint.ElectrostaticConstraint`
    - No big change on Ensemble creation
    - Little increase on computation speed of `mdpy.constraint.CharmmNonbondedConstraint`