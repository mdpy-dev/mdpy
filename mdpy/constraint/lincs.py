import cupy as cp
import numpy as np
from .constraint_base import ConstraintBase
from ._constraint_common import PBC_MIN_IMAGE_DEVICE_FN

# LINCS constraint algorithm (Hess et al., J Chem Theory Comput 2008).
#
# Key concepts and variable naming:
#   coupling_matrix       = the constraint coupling matrix A (off-diagonal
#                           coupling between constraints that share an atom)
#   coupling_denominator  = 1/(1/m_i + 1/m_j) per constraint (the "blc" in LINCS
#                           literature), normalizes each Lagrange multiplier
#   mass_factors          = per-coupled-constraint mass weighting factors
#   coupled_counts/indices = graph of constraints sharing an atom (parallel solve)
#   solution              = the linear-system solution vector (NOT solvent!)
#
# The solver kernel runs these phases:
#   1. Build reference constraint directions from old positions
#   2. Build coupling matrix  A_ij = mass_factor * (rc_i . rc_j)
#   3. Build RHS = coupling_denominator * (rc . delta_new - target_distance)
#   4. Initial solve via Neumann series:  solution = (I + A + ... + A^L) * rhs
#   5. First coordinate update from the Lagrange multipliers
#   6. Centripetal-projection refinement iterations (geometric correction,
#      each with its own embedded Neumann-series solve)

_LINCS_KERNEL = PBC_MIN_IMAGE_DEVICE_FN + r"""
extern "C" __global__
void lincs_kernel(
    const float* __restrict__ old_x,
    const float* __restrict__ old_y,
    const float* __restrict__ old_z,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    const int* __restrict__ constraint_indices,
    const float* __restrict__ target_lengths,
    const float* __restrict__ inverse_mass_i_arr,
    const float* __restrict__ inverse_mass_j_arr,
    const float* __restrict__ coupling_denominator_arr,
    const int* __restrict__ coupled_counts,
    const int* __restrict__ coupled_indices,
    const float* __restrict__ mass_factors,
    float* __restrict__ coupling_matrix,
    int num_constraints,
    int max_coupled,
    int num_constraint_threads,
    int expansion_order,
    int num_iterations
) {
    extern __shared__ float shared_memory[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;

    if (tid >= num_constraint_threads) return;

    int atom_i_sorted = constraint_indices[tid * 2 + 0];
    int atom_j_sorted = constraint_indices[tid * 2 + 1];
    bool is_dummy = (atom_i_sorted < 0);

    // coupling_denominator = 1/(1/m_i + 1/m_j): the shared inverse-mass scale
    // that normalizes each constraint's Lagrange multiplier.
    float target_distance = 0.0f, coupling_denominator = 0.0f, inverse_mass_i = 0.0f, inverse_mass_j = 0.0f;
    float reference_direction_x = 0.0f, reference_direction_y = 0.0f, reference_direction_z = 0.0f;

    if (!is_dummy) {
        target_distance = target_lengths[tid];
        coupling_denominator = coupling_denominator_arr[tid];
        inverse_mass_i = inverse_mass_i_arr[tid];
        inverse_mass_j = inverse_mass_j_arr[tid];
    }

    // Phase 1: reference direction from old positions
    if (!is_dummy) {
        float ox = old_x[atom_i_sorted], oy = old_y[atom_i_sorted], oz = old_z[atom_i_sorted];
        float jx = old_x[atom_j_sorted], jy = old_y[atom_j_sorted], jz = old_z[atom_j_sorted];
        float dx = jx - ox, dy = jy - oy, dz = jz - oz;
        pbc_min_image(dx, dy, dz, pbc_inv, pbc_matrix);
        float inverse_current_distance = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-30f);
        reference_direction_x = dx * inverse_current_distance; reference_direction_y = dy * inverse_current_distance; reference_direction_z = dz * inverse_current_distance;
        shared_memory[lid*3+0] = reference_direction_x; shared_memory[lid*3+1] = reference_direction_y; shared_memory[lid*3+2] = reference_direction_z;
    }
    __syncthreads();

    // Phase 2: build coupling matrix A_ij = mass_factor * (rc_i . rc_j)
    if (!is_dummy) {
        int nc = coupled_counts[tid];
        for (int n = 0; n < nc; n++) {
            int c_idx = coupled_indices[n * num_constraint_threads + tid];
            int c_lid = c_idx - block_offset;
            float mass_factor = mass_factors[n * num_constraint_threads + tid];
            float r1x = shared_memory[c_lid*3+0], r1y = shared_memory[c_lid*3+1], r1z = shared_memory[c_lid*3+2];
            coupling_matrix[n * num_constraint_threads + tid] = mass_factor * (reference_direction_x*r1x + reference_direction_y*r1y + reference_direction_z*r1z);
        }
    }
    __syncthreads();

    // Phase 3: initial RHS = coupling_denominator * (rc . delta_new - target_distance)
    // `solution` is the LINCS linear-system solution vector (NOT solvent).
    float solution = 0.0f;
    if (!is_dummy) {
        float nix = pos_x[atom_i_sorted], niy = pos_y[atom_i_sorted], niz = pos_z[atom_i_sorted];
        float njx = pos_x[atom_j_sorted], njy = pos_y[atom_j_sorted], njz = pos_z[atom_j_sorted];
        float dx = njx - nix, dy = njy - niy, dz = njz - niz;
        pbc_min_image(dx, dy, dz, pbc_inv, pbc_matrix);
        solution = coupling_denominator * (reference_direction_x*dx + reference_direction_y*dy + reference_direction_z*dz - target_distance);
    }

    // Phase 4: Neumann series  solution = (I + A + A^2 + ... + A^L) * rhs
    float* shared_rhs = shared_memory;
    shared_rhs[lid + blockDim.x * 0] = solution;
    __syncthreads();
    for (int rec = 0; rec < expansion_order; rec++) {
        // matrix_vector_product = (A * shared_rhs)[tid] for this expansion step
        float matrix_vector_product = 0.0f;
        if (!is_dummy) {
            int nc = coupled_counts[tid];
            for (int n = 0; n < nc; n++) {
                int c_idx = coupled_indices[n * num_constraint_threads + tid];
                int c_lid = c_idx - block_offset;
                float a_val = coupling_matrix[n * num_constraint_threads + tid];
                matrix_vector_product += a_val * shared_rhs[c_lid + blockDim.x * (rec % 2)];
            }
        }
        shared_rhs[lid + blockDim.x * ((rec+1) % 2)] = matrix_vector_product;
        __syncthreads();
        solution += matrix_vector_product;
    }

    // Phase 5: first coordinate update
    if (!is_dummy) {
        float lagrange = solution;
        float ci = lagrange * inverse_mass_i;
        float cj = -lagrange * inverse_mass_j;
        atomicAdd(&pos_x[atom_i_sorted], reference_direction_x*ci);
        atomicAdd(&pos_y[atom_i_sorted], reference_direction_y*ci);
        atomicAdd(&pos_z[atom_i_sorted], reference_direction_z*ci);
        atomicAdd(&pos_x[atom_j_sorted], reference_direction_x*cj);
        atomicAdd(&pos_y[atom_j_sorted], reference_direction_y*cj);
        atomicAdd(&pos_z[atom_j_sorted], reference_direction_z*cj);
    }
    __syncthreads();

    // Phase 6: centripetal correction iterations
    for (int iter = 0; iter < num_iterations; iter++) {
        float proj = 0.0f;
        if (!is_dummy) {
            float nix = pos_x[atom_i_sorted], niy = pos_y[atom_i_sorted], niz = pos_z[atom_i_sorted];
            float njx = pos_x[atom_j_sorted], njy = pos_y[atom_j_sorted], njz = pos_z[atom_j_sorted];
            float dx = njx - nix, dy = njy - niy, dz = njz - niz;
            pbc_min_image(dx, dy, dz, pbc_inv, pbc_matrix);
            float current_distance_squared = dx*dx + dy*dy + dz*dz;
            // projection_distance_squared = 2*target_distance^2 - |r|^2; the
            // argument whose sqrt appears in the centripetal projection.
            float projection_distance_squared = 2.0f*target_distance*target_distance - current_distance_squared;
            if (projection_distance_squared > 0.0f) {
                proj = coupling_denominator * (target_distance - sqrtf(projection_distance_squared));
            } else {
                proj = coupling_denominator * target_distance;
            }
        }
        float iteration_solution = proj;
        shared_rhs[lid + blockDim.x * 0] = proj;
        __syncthreads();
        for (int rec = 0; rec < expansion_order; rec++) {
            float matrix_vector_product = 0.0f;
            if (!is_dummy) {
                int nc = coupled_counts[tid];
                for (int n = 0; n < nc; n++) {
                    int c_idx = coupled_indices[n * num_constraint_threads + tid];
                    int c_lid = c_idx - block_offset;
                    float a_val = coupling_matrix[n * num_constraint_threads + tid];
                    matrix_vector_product += a_val * shared_rhs[c_lid + blockDim.x * (rec % 2)];
                }
            }
            shared_rhs[lid + blockDim.x * ((rec+1) % 2)] = matrix_vector_product;
            __syncthreads();
            iteration_solution += matrix_vector_product;
        }
        if (!is_dummy) {
            float dl = iteration_solution;
            float ci = dl * inverse_mass_i;
            float cj = -dl * inverse_mass_j;
            atomicAdd(&pos_x[atom_i_sorted], reference_direction_x*ci);
            atomicAdd(&pos_y[atom_i_sorted], reference_direction_y*ci);
            atomicAdd(&pos_z[atom_i_sorted], reference_direction_z*ci);
            atomicAdd(&pos_x[atom_j_sorted], reference_direction_x*cj);
            atomicAdd(&pos_y[atom_j_sorted], reference_direction_y*cj);
            atomicAdd(&pos_z[atom_j_sorted], reference_direction_z*cj);
        }
        __syncthreads();
    }
}
"""


def _build_coupling_data(constraint_pairs, masses, target_lengths, block_size=256):
    num_constraints = len(constraint_pairs)
    if num_constraints == 0:
        e = lambda dt: np.array([], dtype=dt)
        return (e(np.int32), e(np.float32), e(np.float32), e(np.float32),
                e(np.float32), e(np.int32), e(np.int32), e(np.float32),
                e(np.float32), 0, 0)

    atom_to_constraint = {}
    for c, (i, j) in enumerate(constraint_pairs):
        atom_to_constraint.setdefault(i, []).append(c)
        atom_to_constraint.setdefault(j, []).append(c)

    coupled_constraints = [set() for _ in range(num_constraints)]
    for atom, clist in atom_to_constraint.items():
        for a in range(len(clist)):
            for b in range(a + 1, len(clist)):
                coupled_constraints[clist[a]].add(clist[b])
                coupled_constraints[clist[b]].add(clist[a])

    visited = [False] * num_constraints
    groups = []
    for s in range(num_constraints):
        if visited[s]:
            continue
        group = []
        stack = [s]
        while stack:
            c = stack.pop()
            if visited[c]:
                continue
            visited[c] = True
            group.append(c)
            for nb in coupled_constraints[c]:
                if not visited[nb]:
                    stack.append(nb)
        groups.append(group)

    split_map = [0] * num_constraints
    next_slot = 0
    for group in groups:
        group_size = len(group)
        slot_in_block = next_slot % block_size
        if slot_in_block + group_size > block_size:
            next_slot = ((next_slot + block_size - 1) // block_size) * block_size
        for orig in group:
            split_map[orig] = next_slot
            next_slot += 1

    num_constraint_threads = ((next_slot + block_size - 1) // block_size) * block_size

    constraint_indices = np.full((num_constraint_threads, 2), -1, dtype=np.int32)
    inverse_mass_i_array = np.zeros(num_constraint_threads, dtype=np.float32)
    inverse_mass_j_array = np.zeros(num_constraint_threads, dtype=np.float32)
    coupling_denominator_arr = np.zeros(num_constraint_threads, dtype=np.float32)
    target_lengths_array = np.zeros(num_constraint_threads, dtype=np.float32)

    for orig in range(num_constraints):
        slot_index = split_map[orig]
        i, j = constraint_pairs[orig]
        constraint_indices[slot_index, 0] = i
        constraint_indices[slot_index, 1] = j
        inverse_mass_i_array[slot_index] = 1.0 / float(masses[i])
        inverse_mass_j_array[slot_index] = 1.0 / float(masses[j])
        # coupling_denominator = 1/(1/m_i + 1/m_j): the shared inverse-mass scale
        # that normalizes each constraint's Lagrange multiplier.
        coupling_denominator_arr[slot_index] = 1.0 / (inverse_mass_i_array[slot_index] + inverse_mass_j_array[slot_index])
        target_lengths_array[slot_index] = target_lengths[orig]

    max_coupled = 1
    for orig in range(num_constraints):
        max_coupled = max(max_coupled, len(coupled_constraints[orig]))

    coupled_counts = np.zeros(num_constraint_threads, dtype=np.int32)
    coupled_indices = np.zeros(max_coupled * num_constraint_threads, dtype=np.int32)
    mass_factors_array = np.zeros(max_coupled * num_constraint_threads, dtype=np.float32)
    coupling_matrix = np.zeros(max_coupled * num_constraint_threads, dtype=np.float32)

    for orig in range(num_constraints):
        slot_index = split_map[orig]
        i, j = constraint_pairs[orig]
        coupling_denominator_i = coupling_denominator_arr[slot_index]
        coupled = sorted(coupled_constraints[orig])
        coupled_counts[slot_index] = len(coupled)
        for n, c_orig in enumerate(coupled):
            coupled_slot_index = split_map[c_orig]
            ci, cj = constraint_pairs[c_orig]
            coupling_denominator_c = coupling_denominator_arr[coupled_slot_index]
            shared = None
            sign = 1.0
            if i == ci:
                shared = i
                sign = -1.0
            elif j == cj:
                shared = j
                sign = -1.0
            elif i == cj:
                shared = i
                sign = 1.0
            elif j == ci:
                shared = j
                sign = 1.0
            if shared is not None:
                inverse_mass_shared = 1.0 / float(masses[shared])
                mass_factor = sign * inverse_mass_shared * coupling_denominator_i * coupling_denominator_c
                coupled_indices[n * num_constraint_threads + slot_index] = coupled_slot_index
                mass_factors_array[n * num_constraint_threads + slot_index] = mass_factor

    return (constraint_indices.ravel(), target_lengths_array, inverse_mass_i_array, inverse_mass_j_array, coupling_denominator_arr,
            coupled_counts, coupled_indices, mass_factors_array, coupling_matrix,
            num_constraint_threads, max_coupled)


class LincsConstraint(ConstraintBase):
    name = 'lincs'

    def __init__(self, constraint_pairs, target_lengths, masses,
                 expansion_order=4, num_iterations=1):
        self.num_constraints = len(constraint_pairs)
        self.expansion_order = expansion_order
        self.num_iterations = num_iterations

        (constraint_indices, target_lengths_array, inverse_mass_i_array, inverse_mass_j_array, coupling_denominator_arr,
         coupled_counts, coupled_indices, mass_factors_array, coupling_matrix,
         num_constraint_threads, max_coupled) = _build_coupling_data(
            list(constraint_pairs), masses, list(target_lengths))

        self.num_constraint_threads = num_constraint_threads
        self.max_coupled = max_coupled

        self.d_constraint_indices = cp.asarray(constraint_indices)
        self.d_target_lengths = cp.asarray(target_lengths_array)
        self.d_inverse_mass_i = cp.asarray(inverse_mass_i_array)
        self.d_inverse_mass_j = cp.asarray(inverse_mass_j_array)
        self.d_coupling_denominator = cp.asarray(coupling_denominator_arr)
        self.d_coupled_counts = cp.asarray(coupled_counts)
        self.d_coupled_indices = cp.asarray(coupled_indices)
        self.d_mass_factors = cp.asarray(mass_factors_array)
        self.d_coupling_matrix = cp.asarray(coupling_matrix)

        self._num_constraint_indices = constraint_indices.size
        self._kernel = cp.RawKernel(_LINCS_KERNEL, "lincs_kernel")

    def apply(self, gpu_context, time_step, **kwargs):
        if self.num_constraints == 0:
            return
        threads_per_block = 256
        grid = (self.num_constraint_threads + threads_per_block - 1) // threads_per_block
        shared_mem = 3 * threads_per_block * 4
        self._kernel((grid,), (threads_per_block,), (
            gpu_context.d_prev_positions_x,
            gpu_context.d_prev_positions_y,
            gpu_context.d_prev_positions_z,
            gpu_context.d_positions_x,
            gpu_context.d_positions_y,
            gpu_context.d_positions_z,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            self.d_constraint_indices,
            self.d_target_lengths,
            self.d_inverse_mass_i,
            self.d_inverse_mass_j,
            self.d_coupling_denominator,
            self.d_coupled_counts,
            self.d_coupled_indices,
            self.d_mass_factors,
            self.d_coupling_matrix,
            np.int32(self.num_constraints),
            np.int32(self.max_coupled),
            np.int32(self.num_constraint_threads),
            np.int32(self.expansion_order),
            np.int32(self.num_iterations),
        ), shared_mem=shared_mem)
