import pytest

from glyph_mini.pipeline import PipelineConfig, run
from glyph_mini.train import TrainConfig


@pytest.fixture(scope="session")
def small_cfg() -> PipelineConfig:
    """Smaller than the demo so the suite runs in seconds on CPU."""
    return PipelineConfig(n_site_a=6000, n_site_b=4000, sae=TrainConfig(max_steps=1500))


@pytest.fixture(scope="session")
def results(small_cfg):
    return run(small_cfg)
