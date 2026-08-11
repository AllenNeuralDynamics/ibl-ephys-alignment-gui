"""Declarative plot-data registry for ephys plot payloads."""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

PlotMenu = Literal["image", "line", "probe"]
PlotRenderer = Literal["image", "scatter", "line", "probe"]

logger = logging.getLogger(__name__)

_OPTIONAL_PLOT_DEPENDENCIES = frozenset({"brainbox"})
_LOGGED_MISSING_DYNAMIC_DEPENDENCIES: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class PlotSpec:
    """Description of one menu-visible ephys plot payload."""

    key: str
    label: str
    menu: PlotMenu
    renderer: PlotRenderer
    source: Callable[[Any], Any]
    default: bool = False
    bounds_source: Callable[[Any], Any] | None = None
    available: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class DynamicPlotSpec:
    """Description of plot menu entries discovered from payload caches."""

    key: str
    menu: PlotMenu
    children: Callable[[Any], tuple[PlotSpec, ...]]


def _memoized_payload(
    payload_cache: Any,
    key: Hashable,
    build: Callable[[], Any],
) -> Any:
    """Return a memoized typed plot payload when the cache supports it."""
    get_or_build = getattr(payload_cache, "get_or_build_payload", None)
    if callable(get_or_build):
        return get_or_build(key, build)
    return build()


def _source(
    key: Hashable,
    build: Callable[[Any], Any],
    index: int | None = None,
) -> Callable[[Any], Any]:
    """Return a source function backed by typed payload-cache memoization."""

    def source(payload_cache: Any) -> Any:
        value = _memoized_payload(
            payload_cache,
            key,
            lambda: build(payload_cache),
        )
        return value if index is None else value[index]

    return source


def _data_exists(*keys: str) -> Callable[[Any], bool]:
    """Return an availability predicate for ALF data entries."""

    def entry_exists(entry: Any) -> bool:
        if entry is None:
            return False
        get = getattr(entry, "get", None)
        if callable(get):
            return bool(get("exists", False))
        try:
            return bool(entry["exists"])
        except Exception:
            return False

    def available(payload_cache: Any) -> bool:
        data = getattr(payload_cache, "data", None)
        if data is None:
            return True
        for key in keys:
            if not entry_exists(data.get(key)):
                return False
        return True

    return available


def _plot_spec_available(payload_cache: Any, spec: PlotSpec) -> bool:
    if spec.available is None:
        return True
    try:
        return spec.available(payload_cache)
    except Exception:
        logger.warning(
            "Skipping unavailable plot menu entry for %s",
            spec.key,
            exc_info=True,
        )
        return False


def _missing_optional_dependency(exc: Exception) -> str | None:
    """Return optional dependency name when an unavailable plot dependency is missing."""
    if not isinstance(exc, ModuleNotFoundError):
        return None
    name = getattr(exc, "name", None)
    return name if name in _OPTIONAL_PLOT_DEPENDENCIES else None


def _log_dynamic_plot_unavailable(entry_key: str, exc: Exception) -> None:
    """Log unavailable dynamic plot entries without traceback spam for optionals."""
    dependency = _missing_optional_dependency(exc)
    if dependency is not None:
        log_key = (entry_key, dependency)
        if log_key not in _LOGGED_MISSING_DYNAMIC_DEPENDENCIES:
            _LOGGED_MISSING_DYNAMIC_DEPENDENCIES.add(log_key)
            logger.warning(
                "Skipping unavailable dynamic plot menu entries for %s: "
                "optional dependency '%s' is not installed",
                entry_key,
                dependency,
            )
        return

    logger.warning(
        "Skipping unavailable dynamic plot menu entries for %s",
        entry_key,
        exc_info=True,
    )


def _safe_child_key(value: Any) -> str:
    """Return a stable string component for a dynamic plot key."""
    return str(value).replace("/", "_")


def _mapping_child_specs(
    *,
    parent_key: str,
    menu: PlotMenu,
    renderer: PlotRenderer,
    payload: Mapping[Any, Any],
    label: Callable[[Any], str],
    source: Callable[[Any], Mapping[Any, Any]],
    bounds_source: Callable[[Any], Any] | None = None,
) -> tuple[PlotSpec, ...]:
    """Build child specs for a mapping-valued plot payload."""
    return _mapping_child_specs_from_keys(
        parent_key=parent_key,
        menu=menu,
        renderer=renderer,
        child_keys=(
            child_key
            for child_key, child_payload in payload.items()
            if child_payload is not None
        ),
        label=label,
        source=source,
        bounds_source=bounds_source,
    )


def _mapping_child_specs_from_keys(
    *,
    parent_key: str,
    menu: PlotMenu,
    renderer: PlotRenderer,
    child_keys: Iterable[Any],
    label: Callable[[Any], str],
    source: Callable[[Any], Mapping[Any, Any]],
    bounds_source: Callable[[Any], Any] | None = None,
) -> tuple[PlotSpec, ...]:
    """Build child specs from cheaply discovered mapping keys."""
    return tuple(
        PlotSpec(
            key=f"{parent_key}.{_safe_child_key(child_key)}",
            label=label(child_key),
            menu=menu,
            renderer=renderer,
            source=lambda payload_cache, key=child_key: source(payload_cache)[key],
            bounds_source=bounds_source,
        )
        for child_key in child_keys
    )


def _lfp_correlation_payload(payload_cache: Any) -> Mapping[Any, Any]:
    return _memoized_payload(
        payload_cache,
        ("lfp_correlation_data_img",),
        payload_cache.get_lfp_correlation_data_img,
    )


def _lfp_correlation_children(payload_cache: Any) -> tuple[PlotSpec, ...]:
    child_keys = payload_cache.get_lfp_correlation_keys()
    if not child_keys:
        return ()
    return _mapping_child_specs_from_keys(
        parent_key="image.lfp_correlation",
        menu="image",
        renderer="image",
        child_keys=child_keys,
        label=lambda key: f"LFP Correlation ({key})",
        source=_lfp_correlation_payload,
    )


def _passive_events_payload(payload_cache: Any) -> Mapping[Any, Any]:
    return _memoized_payload(
        payload_cache,
        ("passive_events",),
        payload_cache.get_passive_events,
    )


def _passive_event_children(payload_cache: Any) -> tuple[PlotSpec, ...]:
    child_keys = payload_cache.get_passive_event_keys()
    if not child_keys:
        return ()
    return _mapping_child_specs_from_keys(
        parent_key="image.passive_event",
        menu="image",
        renderer="image",
        child_keys=child_keys,
        label=str,
        source=_passive_events_payload,
    )


def _lfp_spectrum_payload(payload_cache: Any, format: str) -> Any:
    return _memoized_payload(
        payload_cache,
        ("lfp_spectrum_data", format),
        lambda: payload_cache.get_lfp_spectrum_data(format),
    )


def _probe_lfp_children(payload_cache: Any) -> tuple[PlotSpec, ...]:
    child_keys = payload_cache.get_lfp_spectrum_probe_keys("lf")
    if not child_keys:
        return ()
    return _mapping_child_specs_from_keys(
        parent_key="probe.lfp_spectrum",
        menu="probe",
        renderer="probe",
        child_keys=child_keys,
        label=str,
        source=lambda payload_cache: _lfp_spectrum_payload(payload_cache, "lf")[1],
    )


def _rfmap_payload(payload_cache: Any) -> Any:
    return _memoized_payload(
        payload_cache,
        ("rfmap_data",),
        payload_cache.get_rfmap_data,
    )


def _rfmap_children(payload_cache: Any) -> tuple[PlotSpec, ...]:
    child_keys = payload_cache.get_rfmap_keys()
    if not child_keys:
        return ()
    return _mapping_child_specs_from_keys(
        parent_key="probe.rfmap",
        menu="probe",
        renderer="probe",
        child_keys=child_keys,
        label=lambda key: f"RF Map - {key}",
        source=lambda payload_cache: _rfmap_payload(payload_cache)[0],
        bounds_source=lambda payload_cache: _rfmap_payload(payload_cache)[1],
    )


PLOT_MENU_ENTRIES: tuple[PlotSpec | DynamicPlotSpec, ...] = (
    PlotSpec(
        key="image.fr",
        label="Firing Rate",
        menu="image",
        renderer="image",
        source=_source(("fr_img",), lambda payload_cache: payload_cache.get_fr_img()),
        default=True,
        available=_data_exists("spikes"),
    ),
    PlotSpec(
        key="scatter.amplitude",
        label="Amplitude",
        menu="image",
        renderer="scatter",
        source=_source(
            ("depth_data_scatter",),
            lambda payload_cache: payload_cache.get_depth_data_scatter(),
        ),
        available=_data_exists("spikes"),
    ),
    PlotSpec(
        key="image.spike_correlation",
        label="Spike Correlation",
        menu="image",
        renderer="image",
        source=_source(
            ("spike_correlation_data_img",),
            lambda payload_cache: payload_cache.get_spike_correlation_data_img(),
        ),
        available=_data_exists("spikes"),
    ),
    PlotSpec(
        key="image.rms_ap",
        label="RMS AP",
        menu="image",
        renderer="image",
        source=_source(
            ("rms_data_img_probe", "AP"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("AP"),
            0,
        ),
        available=_data_exists("rms_AP"),
    ),
    PlotSpec(
        key="image.rms_ap_main",
        label="RMS AP Main Rec",
        menu="image",
        renderer="image",
        source=_source(
            ("rms_data_img_probe", "AP_main"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("AP_main"),
            0,
        ),
        available=_data_exists("rms_AP_main"),
    ),
    PlotSpec(
        key="image.rms_lfp",
        label="RMS LFP",
        menu="image",
        renderer="image",
        source=_source(
            ("rms_data_img_probe", "LF"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("LF"),
            0,
        ),
        available=_data_exists("rms_LF"),
    ),
    PlotSpec(
        key="image.rms_lfp_main",
        label="RMS LFP Main Rec",
        menu="image",
        renderer="image",
        source=_source(
            ("rms_data_img_probe", "LF_main"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("LF_main"),
            0,
        ),
        available=_data_exists("rms_LF_main"),
    ),
    PlotSpec(
        key="image.lfp_spectrum",
        label="LFP Spectrum",
        menu="image",
        renderer="image",
        source=_source(
            ("lfp_spectrum_data", "lf"),
            lambda payload_cache: payload_cache.get_lfp_spectrum_data("lf"),
            0,
        ),
        available=_data_exists("psd_lf"),
    ),
    PlotSpec(
        key="image.lfp_spectrum_main",
        label="LFP Spectrum Main Rec",
        menu="image",
        renderer="image",
        source=_source(
            ("lfp_spectrum_data", "lf_main"),
            lambda payload_cache: payload_cache.get_lfp_spectrum_data("lf_main"),
            0,
        ),
        available=_data_exists("psd_lf_main"),
    ),
    DynamicPlotSpec(
        key="image.lfp_correlation",
        menu="image",
        children=_lfp_correlation_children,
    ),
    PlotSpec(
        key="scatter.cluster_fr",
        label="Cluster Amp vs Depth vs FR",
        menu="image",
        renderer="scatter",
        source=_source(
            ("fr_p2t_data_scatter",),
            lambda payload_cache: payload_cache.get_fr_p2t_data_scatter(),
            0,
        ),
        available=_data_exists("spikes", "clusters"),
    ),
    PlotSpec(
        key="scatter.cluster_duration",
        label="Cluster Amp vs Depth vs Duration",
        menu="image",
        renderer="scatter",
        source=_source(
            ("fr_p2t_data_scatter",),
            lambda payload_cache: payload_cache.get_fr_p2t_data_scatter(),
            1,
        ),
        available=_data_exists("spikes", "clusters"),
    ),
    PlotSpec(
        key="scatter.cluster_amp",
        label="Cluster FR vs Depth vs Amp",
        menu="image",
        renderer="scatter",
        source=_source(
            ("fr_p2t_data_scatter",),
            lambda payload_cache: payload_cache.get_fr_p2t_data_scatter(),
            2,
        ),
        available=_data_exists("spikes", "clusters"),
    ),
    DynamicPlotSpec(
        key="image.passive_event",
        menu="image",
        children=_passive_event_children,
    ),
    PlotSpec(
        key="line.fr",
        label="Firing Rate",
        menu="line",
        renderer="line",
        source=_source(
            ("fr_amp_data_line",),
            lambda payload_cache: payload_cache.get_fr_amp_data_line(),
            0,
        ),
        default=True,
        available=_data_exists("spikes"),
    ),
    PlotSpec(
        key="line.amplitude",
        label="Amplitude",
        menu="line",
        renderer="line",
        source=_source(
            ("fr_amp_data_line",),
            lambda payload_cache: payload_cache.get_fr_amp_data_line(),
            1,
        ),
        available=_data_exists("spikes"),
    ),
    PlotSpec(
        key="probe.rms_ap",
        label="RMS AP",
        menu="probe",
        renderer="probe",
        source=_source(
            ("rms_data_img_probe", "AP"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("AP"),
            1,
        ),
        default=True,
        available=_data_exists("rms_AP"),
    ),
    PlotSpec(
        key="probe.rms_lfp",
        label="RMS LFP",
        menu="probe",
        renderer="probe",
        source=_source(
            ("rms_data_img_probe", "LF"),
            lambda payload_cache: payload_cache.get_rms_data_img_probe("LF"),
            1,
        ),
        available=_data_exists("rms_LF"),
    ),
    DynamicPlotSpec(
        key="probe.lfp_spectrum",
        menu="probe",
        children=_probe_lfp_children,
    ),
    DynamicPlotSpec(
        key="probe.rfmap",
        menu="probe",
        children=_rfmap_children,
    ),
)

PLOT_SPECS: tuple[PlotSpec, ...] = tuple(
    entry for entry in PLOT_MENU_ENTRIES if isinstance(entry, PlotSpec)
)

_PLOT_SPEC_BY_KEY = {spec.key: spec for spec in PLOT_SPECS}


def plot_spec(key: str) -> PlotSpec:
    """Return a plot spec by stable key."""
    return _PLOT_SPEC_BY_KEY[key]


def plot_specs_for_menu(menu: PlotMenu) -> tuple[PlotSpec, ...]:
    """Return plot specs shown in one menu group, in menu order."""
    return tuple(spec for spec in PLOT_SPECS if spec.menu == menu)


def available_plot_specs_for_menu(
    payload_cache: Any,
    menu: PlotMenu,
) -> tuple[PlotSpec, ...]:
    """Return menu specs available for the current plot payload cache."""
    specs: list[PlotSpec] = []
    for entry in PLOT_MENU_ENTRIES:
        if entry.menu != menu:
            continue
        if isinstance(entry, PlotSpec):
            if _plot_spec_available(payload_cache, entry):
                specs.append(entry)
            continue
        try:
            specs.extend(entry.children(payload_cache))
        except Exception as exc:
            _log_dynamic_plot_unavailable(entry.key, exc)
    return tuple(specs)


def default_plot_spec(menu: PlotMenu) -> PlotSpec:
    """Return the default plot spec for a menu group."""
    for spec in plot_specs_for_menu(menu):
        if spec.default:
            return spec
    raise KeyError(f"No default plot spec for menu {menu!r}")


def _coerce_spec(spec_or_key: PlotSpec | str) -> PlotSpec:
    if isinstance(spec_or_key, PlotSpec):
        return spec_or_key
    return plot_spec(spec_or_key)


def resolve_plot_payload(payload_cache: Any, spec_or_key: PlotSpec | str) -> Any:
    """Resolve a plot payload from a cache using a plot spec key."""
    spec = _coerce_spec(spec_or_key)
    try:
        return spec.source(payload_cache)
    except Exception:
        logger.warning("Plot payload %s is unavailable", spec.key, exc_info=True)
        return None


def resolve_plot_bounds(payload_cache: Any, spec_or_key: PlotSpec | str) -> Any:
    """Resolve optional plot bounds from a cache using a plot spec key."""
    spec = _coerce_spec(spec_or_key)
    if spec.bounds_source is None:
        return None
    try:
        return spec.bounds_source(payload_cache)
    except Exception:
        logger.warning("Plot bounds %s are unavailable", spec.key, exc_info=True)
        return None


def mapping_plot_specs(
    *,
    parent_key: str,
    menu: PlotMenu,
    renderer: PlotRenderer,
    payloads: Mapping[Any, Any],
    label: Callable[[Any], str] = str,
) -> tuple[PlotSpec, ...]:
    """Build specs from an already-available payload mapping."""
    return _mapping_child_specs(
        parent_key=parent_key,
        menu=menu,
        renderer=renderer,
        payload=payloads,
        label=label,
        source=lambda _payload_cache: payloads,
    )
