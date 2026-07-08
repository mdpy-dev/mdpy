"""Internal CUDA device-function snippets shared across the constraint kernels.

Currently holds the minimum-image PBC helper used by both LINCS and SETTLE.
Prepended to each constraint kernel's source string so the kernel can call
``pbc_min_image(...)``.

Note: the bonded-force subsystem (``mdpy.force.bonded_force`` /
``bonded_transpiler``) uses its own float3-typed ``pbc_wrap_vec`` variant
inside its ``_PREAMBLE``; the two intentionally live with their respective
subsystems (different vector conventions: float& here vs float3 there) rather
than being unified, to keep force/ and constraint/ free of cross-subsystem
dependencies.
"""

PBC_MIN_IMAGE_DEVICE_FN = r"""
__device__ __forceinline__ void pbc_min_image(
    float& dx, float& dy, float& dz,
    const float* __restrict__ pbc_inv,
    const float* __restrict__ pbc_matrix
) {
    float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
    float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
    float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
    fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
    dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
    dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
    dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
}
"""
