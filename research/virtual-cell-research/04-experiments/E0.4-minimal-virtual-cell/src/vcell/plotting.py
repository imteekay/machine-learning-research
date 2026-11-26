"""One quiet, consistent figure style for vcell, in light and dark.

The colors come from the validated reference palette of the dataviz guide, checked for colorblind
separation in both modes:

- Slot 1 blue (and slot 2 orange when a second identity is needed) carries the series.
- Muted gray is for context. The default form is *emphasis*: the model under test is blue and
  everything else is gray.

Text never wears a series color. Gridlines are solid hairlines.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "blue": "#3987e5",
        "orange": "#d95926",
    },
}


def rc_params(name: str) -> dict:
    t = THEMES[name]
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["text"],
        "axes.titlecolor": t["text"],
        "axes.labelcolor": t["text2"],
        "xtick.labelcolor": t["text2"],
        "ytick.labelcolor": t["text2"],
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.75,
        "grid.linestyle": "-",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


@contextmanager
def theme(name: str = "light") -> Iterator[dict[str, str]]:
    """Use a theme for the figures created inside the block. Yields its color tokens."""
    with mpl.rc_context(rc_params(name)):
        yield THEMES[name]


def save_light_dark(draw: Callable[[dict[str, str]], plt.Figure], stem: Path, dpi: int = 200) -> list[Path]:
    """Render ``draw(tokens)`` in both themes and save ``<stem>-light.png`` and ``<stem>-dark.png``."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("light", "dark"):
        with theme(name) as tokens:
            fig = draw(tokens)
            path = stem.with_name(f"{stem.name}-{name}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
            plt.close(fig)
            paths.append(path)
    return paths
