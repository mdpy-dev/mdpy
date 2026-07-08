class ConstraintBase:
    name: str = ''

    def apply(self, state, time_step):
        raise NotImplementedError
