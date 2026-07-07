class ConstraintBase:
    name: str = ''

    def apply(self, gpu_context, time_step):
        raise NotImplementedError
