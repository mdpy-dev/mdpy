from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PMEParameters:
    alpha: float
    grid_x: int
    grid_y: int
    grid_z: int
    order: int = 4

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
        tolerance: float = 1e-5,
        order: int = 4,
    ) -> PMEParameters:
        alpha = math.sqrt(-math.log(tolerance)) / cutoff

        def grid_dim(box_dim: float) -> int:
            raw = 2.0 * alpha * box_dim / (3.0 * tolerance ** 0.2)
            return cls._next_fft_friendly_size(max(order, int(math.ceil(raw))))

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
