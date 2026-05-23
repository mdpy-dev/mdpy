from __future__ import annotations

import math
from numba import cuda


@cuda.jit(device=True)
def minimum_image(delta_x, delta_y, delta_z, pbc_matrix, pbc_inv):
    sx = delta_x * pbc_inv[0] + delta_y * pbc_inv[3] + delta_z * pbc_inv[6]
    sy = delta_x * pbc_inv[1] + delta_y * pbc_inv[4] + delta_z * pbc_inv[7]
    sz = delta_x * pbc_inv[2] + delta_y * pbc_inv[5] + delta_z * pbc_inv[8]
    sx -= round(sx)
    sy -= round(sy)
    sz -= round(sz)
    out_x = sx * pbc_matrix[0] + sy * pbc_matrix[3] + sz * pbc_matrix[6]
    out_y = sx * pbc_matrix[1] + sy * pbc_matrix[4] + sz * pbc_matrix[7]
    out_z = sx * pbc_matrix[2] + sy * pbc_matrix[5] + sz * pbc_matrix[8]
    return out_x, out_y, out_z
