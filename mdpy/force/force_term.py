class ForceTerm:

    name: str = ''

    def compute(self, state, block_list=None, compute_energy=True):
        raise NotImplementedError
