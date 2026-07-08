class ForceTerm:

    name: str = ''

    def compute(self, state, block_list=None, compute_energy=True):
        raise NotImplementedError

    def __add__(self, other):
        if not isinstance(other, ForceTerm):
            return NotImplemented
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(other).__name__}"
            )
        from mdpy.force.force_group import ForceGroup
        return ForceGroup([self, other])
