class BarostatBase:
    name: str = ''

    def apply(self, system):
        raise NotImplementedError
