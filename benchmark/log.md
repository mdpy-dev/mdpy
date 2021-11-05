# Profile Log

## 00_cpu_origin

- Date: 2021-11-04
- Modification: None
- Conclusion:
    - Slow

## 01_cpu_numba_njit

- Date: 2021-11-04
- Modification:
    - Add kernel function to each constraint class
    - Add njit to each kernel
- Conclusion:
    - 7x faster to 00 version

## 02_cpu_numba_internal_njit

- Date: 2021-11-05
- Modification:
    - Replace the kernel function with a constraint.kernel method
    - Compile jit(kernel) in the initialization of each constraint
- Conclusion:
    - No big change on computational speed
    - Decreasing time of `import mdpy as md`
    - Supporting user' specification of environment variables

## 03_cpu_numba_electrostatic_io

- Date: 2021-11-05
- Modification:
    - Update the bind_ensemble method of mdpy.constraint.ElectrostaticConstraint, removing pair-wise parameter
- Conclusion
    - The IO time of mdpy.constraint.ElectrostaticConstraint decreases from 19s to 0.5s