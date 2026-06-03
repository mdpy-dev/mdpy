from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import erfc


@dataclass
class PMEParameters:
    alpha: float
    grid_x: int
    grid_y: int
    grid_z: int
    order: int = 4

    @staticmethod
    def _calc_ewald_coefficient(cutoff: float, rtol: float = 1e-5) -> float:
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if erfc(mid * cutoff) > rtol:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    @staticmethod
    def _next_fft_friendly_size(n: int) -> int:
        while True:
            m = n
            for p in (7, 5, 3, 2):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1

    @classmethod
    def from_box(
        cls,
        box_x: float,
        box_y: float,
        box_z: float,
        cutoff: float,
        order: int = 4,
        ewald_rtol: float = 1e-5,
        fourier_spacing: float = 1.2,
    ) -> PMEParameters:
        alpha = cls._calc_ewald_coefficient(cutoff, ewald_rtol)

        def grid_dim(box_dim: float) -> int:
            nmin = max(order, math.ceil(box_dim / fourier_spacing))
            return cls._next_fft_friendly_size(nmin)

        return cls(
            alpha=alpha,
            grid_x=grid_dim(box_x),
            grid_y=grid_dim(box_y),
            grid_z=grid_dim(box_z),
            order=order,
        )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (self.grid_x, self.grid_y, self.grid_z)
