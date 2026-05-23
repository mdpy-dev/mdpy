import pytest

test_order = [
    'new_particle_table',
    'new_pbc',
    'new_topology',
    'new_force_term',
    'bonded_force',
    'tile_list',
    'system',
    'charmm_forcefield',
]

def pytest_collection_modifyitems(items):
    current_index = 0
    for test in test_order:
        indexes = []
        for id, item in enumerate(items):
            if 'test_'+test+'.py' in item.nodeid:
                indexes.append(id)
        for id, index in enumerate(indexes):
            items[current_index+id], items[index] = items[index], items[current_index+id]
        current_index += len(indexes)
