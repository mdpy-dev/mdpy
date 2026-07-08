from mdpy.force.force_term import ForceTerm


def test_force_term_interface():
    term = ForceTerm()
    assert term.name == ''
    try:
        term.compute(None)
        assert False, 'should raise'
    except NotImplementedError:
        pass


def test_force_term_subclass():
    class MyForce(ForceTerm):
        name = 'my_force'

        def compute(self, state, block_list=None):
            return 42.0

    term = MyForce()
    assert term.name == 'my_force'
    assert term.compute(None) == 42.0
