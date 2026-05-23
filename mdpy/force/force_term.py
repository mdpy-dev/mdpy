class ForceTerm:

    name: str = ''

    def compute(self, gpu_context, tile_list=None):
        raise NotImplementedError
