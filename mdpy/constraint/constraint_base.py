class ConstraintBase:
    name: str = ''

    def apply(self, gpu_context, time_step):
        raise NotImplementedError

    def remap_indices_gpu(self, d_remap):
        raise NotImplementedError
