"""Qt-free raster sizing requests for image-like plot payloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRasterRequest:
    """Requested raster resolution for a viewer-backed image payload."""

    max_time_bins: int
    max_depth_bins: int
    min_time_bin_s: float = 0.05
    min_depth_bin_um: float = 5.0

    @classmethod
    def from_plot_size(
        cls,
        *,
        width_px: float,
        height_px: float,
    ) -> ImageRasterRequest:
        """Build a request from the displayed plot area in logical pixels."""
        return cls(
            max_time_bins=max(1, int(round(width_px))),
            max_depth_bins=max(1, int(round(height_px))),
        )

    def time_bin_s(self, duration_s: float) -> float:
        """Return a time bin that does not exceed the requested raster width."""
        return max(duration_s / self.max_time_bins, self.min_time_bin_s)

    def depth_bin_um(self, depth_span_um: float) -> float:
        """Return a depth bin that does not exceed the requested raster height."""
        return max(depth_span_um / self.max_depth_bins, self.min_depth_bin_um)


DEFAULT_IMAGE_RASTER_REQUEST = ImageRasterRequest(
    max_time_bins=800,
    max_depth_bins=600,
)
