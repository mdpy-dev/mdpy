"""3D Hilbert curve encoder.

Produces the spatial key used for intra-cell sorting in the block list
(Phase 3). A Hilbert curve of order ``L`` maps each cell of the
``2^L x 2^L x 2^L`` grid to a unique index in ``[0, 2^(3L))`` such that
consecutive indices are always *face-adjacent* cells (they differ in exactly
one coordinate by one). This face-adjacency is what gives tighter block AABBs
than a Morton (Z-order) key, where consecutive cells can share only an edge or
a vertex.

The encoder is the cubic specialisation of Hamilton's compact Hilbert index
algorithm (Hamilton, "Compact Hilbert Indices", Dalhousie CS-2006-07, Alg. 7).
For a cubic grid every level mask is all-ones, so the algorithm reduces to a
single per-level loop with three N=3 lookup tables. The correctness gate is the
face-adjacency test in ``test_block_list_minimization.py`` (verified for
L = 1, 2, 3).

GPU side: ``HILBERT_ENCODE_KERNEL`` exposes the same algorithm as a CUDA device
function for use by the ``cell_assign`` kernel (Task 3.2). The lookup tables are
identical to the host ones.
"""
from __future__ import annotations

# --- N = 3 lookup tables (Hamilton CS-2006-07) ------------------------------
# These are derived, not magic: see module docstring.
#
# ``_INVERSE_GC[w]``  = inverse Gray code of w  (w in [0, 8)).
# ``_ENTRY[w]``       = entry point of hypercube w.
# ``_DIR[w]``         = axis along which sub-curve w progresses (0=x,1=y,2=z).
_INVERSE_GC = (0, 1, 3, 2, 7, 6, 4, 5)
_ENTRY = (0, 0, 0, 3, 3, 6, 6, 5)
_DIR = (0, 1, 1, 2, 2, 1, 1, 0)

_MASK3 = 0x7  # keep values inside 3 bits


def _rotate_right3(value: int, count: int) -> int:
    """Rotate the low 3 bits of ``value`` right by ``count`` (mod 3) places."""
    count &= 3
    return ((value >> count) | (value << (3 - count))) & _MASK3


def _rotate_left3(value: int, count: int) -> int:
    """Rotate the low 3 bits of ``value`` left by ``count`` (mod 3) places."""
    count &= 3
    return ((value << count) | (value >> (3 - count))) & _MASK3


def hilbert_index(x: int, y: int, z: int, L: int) -> int:
    """Encode 3D coordinates into a Hilbert curve index.

    Args:
        x, y, z: integer coordinates, each in ``[0, 2^L)``.
        L: number of levels (``B = 3 * L`` bits total; ``2^L`` cells per axis).

    Returns:
        int in ``[0, 2^(3*L))`` -- the Hilbert index. Consecutive indices are
        guaranteed face-adjacent (the defining Hilbert property).
    """
    h = 0
    ve = 0          # "ve": orientation entry pattern, accumulated across levels
    vd = 2          # "vd": rotation direction, starts at N - 1 (= 2 for N = 3)
    for i in range(L - 1, -1, -1):
        # 1. gather the 3 bits of this level into a single digit l = (z y x)
        level = (((x >> i) & 1)
                 | (((y >> i) & 1) << 1)
                 | (((z >> i) & 1) << 2))
        # 2. orient the digit with the current rotation state
        level = _rotate_right3(level ^ ve, (vd + 1) % 3)
        # 3. inverse-Gray-code -> the Hilbert sub-curve index at this level
        w = _INVERSE_GC[level]
        # 4. append 3 bits to the running Hilbert index (MSB first)
        h = (h << 3) | w
        # 5. update orientation state for the next (finer) level
        ve ^= _rotate_left3(_ENTRY[w], (vd + 1) % 3)
        vd = (vd + _DIR[w] + 1) % 3
    return h


# --- GPU mirror -------------------------------------------------------------
# CUDA C device functions mirroring ``hilbert_index`` exactly. The lookup tables
# are identical to the host tuples above. Intended to be concatenated into the
# ``cell_assign`` kernel source (Task 3.2); not launched in this task.
HILBERT_ENCODE_KERNEL = r"""
__constant__ unsigned int HILBERT_INVERSE_GC[8] = {0, 1, 3, 2, 7, 6, 4, 5};
__constant__ unsigned int HILBERT_ENTRY[8]      = {0, 0, 0, 3, 3, 6, 6, 5};
__constant__ unsigned int HILBERT_DIR[8]        = {0, 1, 1, 2, 2, 1, 1, 0};

__device__ __forceinline__
unsigned int hilbert_rotate_right3(unsigned int v, unsigned int r) {
    r &= 3u;
    return ((v >> r) | (v << (3u - r))) & 7u;
}

__device__ __forceinline__
unsigned int hilbert_rotate_left3(unsigned int v, unsigned int r) {
    r &= 3u;
    return ((v << r) | (v >> (3u - r))) & 7u;
}

/* Encode a cell-local coordinate (lx, ly, lz) in [0, 2^L) into a Hilbert index
 * in [0, 2^(3L)). Mirrors the host mdpy.core.hilbert.hilbert_index. */
__device__ __forceinline__
unsigned int hilbert_encode(unsigned int lx, unsigned int ly, unsigned int lz, int L) {
    unsigned int h = 0u;
    unsigned int ve = 0u;
    unsigned int vd = 2u;  /* N - 1 for N = 3 */
    for (int i = L - 1; i >= 0; i--) {
        unsigned int level = ((lx >> i) & 1u)
                           | (((ly >> i) & 1u) << 1)
                           | (((lz >> i) & 1u) << 2);
        unsigned int rot = (vd + 1u) % 3u;
        level = hilbert_rotate_right3(level ^ ve, rot);
        unsigned int w = HILBERT_INVERSE_GC[level];
        h = (h << 3) | w;
        ve ^= hilbert_rotate_left3(HILBERT_ENTRY[w], rot);
        vd = (vd + HILBERT_DIR[w] + 1u) % 3u;
    }
    return h;
}
"""
