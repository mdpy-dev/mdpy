class ConstraintBase:
    name: str = ''

    def apply(self, gpu_context, dt):
        raise NotImplementedError

    def remap_indices_gpu(self, d_remap, d_rebuild_flag):
        raise NotImplementedError
