class ForceTerm:

    name: str = ''

    def compute(self, gpu_context, block_list=None):
        raise NotImplementedError
