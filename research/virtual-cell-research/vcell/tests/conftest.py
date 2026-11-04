import pytest

from vcell.data import build_perturbation_data
from vcell.simulate import ToyCell, ToyCellConfig

TINY = ToyCellConfig(
    n_genes=50,
    n_regulators=10,
    n_masters=3,
    n_stress_genes=5,
    n_null_perts=5,
    n_ctrl=200,
    cells_per_pert=20,
    oracle_cells=300,
)


@pytest.fixture(scope="session")
def tiny_cell() -> ToyCell:
    return ToyCell(TINY, seed=7)


@pytest.fixture(scope="session")
def tiny_screen(tiny_cell):
    return tiny_cell.simulate_screen(seed=1)


@pytest.fixture(scope="session")
def tiny_data(tiny_cell, tiny_screen):
    return build_perturbation_data(tiny_screen, strata=tiny_cell.perturbation_strata)
