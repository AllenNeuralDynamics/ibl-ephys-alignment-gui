"""Plan runtime dependencies required to save edited alignments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ephys_alignment_gui.core.document import AlignmentDocument, AlignmentKey
from ephys_alignment_gui.io.alignment_data_context import AlignmentDataContext
from ephys_alignment_gui.io.datapackage_loader import MouseRoot, ProbeInfo
from ephys_alignment_gui.io.load_data_target import LoadDataJobTarget
from ephys_alignment_gui.runtime.ephys_stream import StreamKey
from ephys_alignment_gui.runtime.session import SessionRuntime
from ephys_alignment_gui.services.ephys_data import ChannelTable

SaveRuntimeStatus = Literal["active", "cached", "missing", "unresolvable"]


@dataclass(frozen=True)
class SaveRuntimeDependency:
    """Runtime and metadata dependency for saving one edited alignment."""

    key: AlignmentKey
    stream_key: StreamKey
    status: SaveRuntimeStatus
    stream_runtime: Any | None = None
    mouse_root: MouseRoot | None = None
    probe: ProbeInfo | None = None
    channel_table: ChannelTable | None = None
    load_target: LoadDataJobTarget | None = None
    message: str | None = None

    @property
    def available(self) -> bool:
        """Whether runtime-derived save data is available now."""
        return self.stream_runtime is not None

    @property
    def needs_reload(self) -> bool:
        """Whether save needs this runtime to be reloaded before it can proceed."""
        return self.status == "missing"

    @property
    def unresolvable(self) -> bool:
        """Whether metadata is insufficient to address this save dependency."""
        return self.status == "unresolvable"


@dataclass(frozen=True)
class SaveRuntimeDependencyPlan:
    """Plan for the runtime dependencies of dirty document alignments."""

    dependencies: tuple[SaveRuntimeDependency, ...]

    @property
    def by_key(self) -> dict[AlignmentKey, SaveRuntimeDependency]:
        """Return dependencies keyed by alignment key."""
        return {dependency.key: dependency for dependency in self.dependencies}

    @property
    def unavailable(self) -> tuple[SaveRuntimeDependency, ...]:
        """Dependencies that cannot be saved with currently loaded runtime data."""
        return tuple(
            dependency for dependency in self.dependencies if not dependency.available
        )

    @property
    def eviction_protected(self) -> tuple[SaveRuntimeDependency, ...]:
        """Available dirty dependencies that would be harmed by cache eviction."""
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.available and dependency.status in {"active", "cached"}
        )

    def failure_message(self) -> str | None:
        """Return a user-facing failure message for the first unavailable item."""
        if not self.unavailable:
            return None
        return self.unavailable[0].message


def plan_save_runtime_dependencies(
    *,
    document: AlignmentDocument,
    data_context: AlignmentDataContext,
    runtime: SessionRuntime,
) -> SaveRuntimeDependencyPlan:
    """Build a read-only plan for saving all dirty document alignments."""
    dependencies = [
        _dependency_for_key(key, data_context=data_context, runtime=runtime)
        for key in _sorted_keys(document.dirty_alignment_states())
    ]
    return SaveRuntimeDependencyPlan(tuple(dependencies))


def describe_save_runtime_dependencies(
    dependencies: Iterable[SaveRuntimeDependency],
) -> str:
    """Return a compact description of save runtime dependencies."""
    return ", ".join(
        f"{dependency.key.recording_id}/{dependency.key.ephys_collection} "
        f"shank {dependency.key.shank_idx + 1}"
        for dependency in dependencies
    )


def _dependency_for_key(
    key: AlignmentKey,
    *,
    data_context: AlignmentDataContext,
    runtime: SessionRuntime,
) -> SaveRuntimeDependency:
    stream_key = (key.recording_id, key.ephys_collection)
    stream_runtime, status = _available_runtime_for_key(key, runtime)
    mouse_root = data_context.mouse_root
    if mouse_root is None:
        return SaveRuntimeDependency(
            key=key,
            stream_key=stream_key,
            status=status if stream_runtime is not None else "unresolvable",
            stream_runtime=stream_runtime,
            message=(
                "Cannot save edited alignment for "
                f"{key.recording_id}/{key.ephys_collection} shank "
                f"{key.shank_idx + 1}: no mouse root is loaded."
            ),
        )

    try:
        probe = data_context.probe_for_stream_key(
            key.recording_id,
            key.ephys_collection,
        )
    except Exception as exc:
        return SaveRuntimeDependency(
            key=key,
            stream_key=stream_key,
            status=status if stream_runtime is not None else "unresolvable",
            stream_runtime=stream_runtime,
            mouse_root=mouse_root,
            message=(
                "Cannot resolve runtime needed to save edited alignment for "
                f"{key.recording_id}/{key.ephys_collection} shank "
                f"{key.shank_idx + 1}: {exc}"
            ),
        )

    channel_table = _channel_table_for_dependency(
        stream_runtime=stream_runtime,
        data_context=data_context,
        probe=probe,
    )
    load_target = _load_target_for_dependency(
        key=key,
        stream_key=stream_key,
        mouse_root=mouse_root,
        probe=probe,
        channel_table=channel_table,
    )

    if stream_runtime is not None:
        return SaveRuntimeDependency(
            key=key,
            stream_key=stream_key,
            status=status,
            stream_runtime=stream_runtime,
            mouse_root=mouse_root,
            probe=probe,
            channel_table=channel_table,
            load_target=load_target,
        )

    return SaveRuntimeDependency(
        key=key,
        stream_key=stream_key,
        status="missing",
        mouse_root=mouse_root,
        probe=probe,
        channel_table=channel_table,
        load_target=load_target,
        message=(
            "Cannot save edited alignment for "
            f"{key.recording_id}/{key.ephys_collection} shank "
            f"{key.shank_idx + 1}: stream runtime is not loaded."
        ),
    )


def _available_runtime_for_key(
    key: AlignmentKey,
    runtime: SessionRuntime,
) -> tuple[Any | None, SaveRuntimeStatus]:
    stream_key = (key.recording_id, key.ephys_collection)
    active = runtime.active_stream_runtime
    if active is not None:
        active_stream_key = getattr(active, "stream_key", None)
        if active_stream_key == stream_key or (
            active_stream_key is None and runtime.current_stream_key == stream_key
        ):
            return active, "active"

    cached = runtime.cached_stream(stream_key)
    if cached is not None:
        return cached, "cached"

    return None, "missing"


def _channel_table_for_dependency(
    *,
    stream_runtime: Any | None,
    data_context: AlignmentDataContext,
    probe: ProbeInfo,
) -> ChannelTable | None:
    if stream_runtime is not None:
        stream = getattr(stream_runtime, "stream", None)
        channel_table = getattr(stream, "channel_table", None)
        if channel_table is not None:
            return channel_table

    active_probe = data_context.probe_info
    if (
        active_probe is not None
        and active_probe.recording_id == probe.recording_id
        and active_probe.ephys_collection == probe.ephys_collection
    ):
        return data_context.channel_table

    return None


def _load_target_for_dependency(
    *,
    key: AlignmentKey,
    stream_key: StreamKey,
    mouse_root: MouseRoot,
    probe: ProbeInfo,
    channel_table: ChannelTable | None,
) -> LoadDataJobTarget | None:
    if channel_table is None:
        return None
    return LoadDataJobTarget(
        recording_id=key.recording_id,
        probe_name=probe.probe_name,
        stream_key=stream_key,
        shank_idx=key.shank_idx,
        mouse_root=mouse_root,
        probe_info=probe,
        channel_table=channel_table,
    )


def _sorted_keys(states: dict[AlignmentKey, object]) -> list[AlignmentKey]:
    return sorted(
        states,
        key=lambda key: (
            key.recording_id,
            key.ephys_collection,
            key.shank_idx,
        ),
    )
