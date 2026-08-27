"""Cyclic colour mapping for complex coherency (phase and magnitude)."""

from functools import lru_cache

import numpy as np

_LUT_N = 256

# Measured-but-incoherent tone: light enough to read as a plinth under the data,
# dark enough to separate from the white page, which means "never measured".
MEASURED_FLOOR = np.array([0.876, 0.876, 0.886])


@lru_cache(maxsize=1)
def phase_lut() -> np.ndarray:
    """``(_LUT_N, 3)`` float RGB ring covering phase over one turn.

    cmocean's ``phase`` is cyclic and near-isoluminant (L* 51-56), so hue
    carries phase at a near-constant perceptual rate and lightness is left
    free to carry magnitude. An HSV wheel is neither: its perceptual step
    rate varies ~20x across the turn, which is what makes green look like a
    plateau and yellow like a knife edge.
    """
    from cmocean import cm as cmo

    ring = np.asarray(cmo.phase(np.linspace(0.0, 1.0, _LUT_N, endpoint=False)))
    return np.ascontiguousarray(ring[:, :3], dtype=np.float64)


def phase_magnitude_rgb(
    phase: np.ndarray,
    magnitude: np.ndarray,
    floor: np.ndarray = MEASURED_FLOOR,
) -> np.ndarray:
    """Float RGB for phase (radians) faded towards ``floor``.

    ``magnitude`` is expected pre-normalised to [0, 1]. Zero lands on the floor
    rather than on the page, so an incoherent pair inside a measured block
    reads as part of the block instead of a hole punched through to the
    background.
    """
    lut = phase_lut()
    idx = (((np.asarray(phase) / (2 * np.pi)) % 1.0) * _LUT_N).astype(int) % _LUT_N
    rgb = lut[idx]
    weight = np.clip(np.asarray(magnitude, dtype=float), 0.0, 1.0)[..., None]
    return floor + weight * (rgb - floor)
