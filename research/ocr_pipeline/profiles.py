"""The profile table: which engine feeds which tree builder, and with what merge policy.

A profile is the unit a comparison run is named by. It fixes the engine that
produced the artifact, the builder that turns it into a tree, and whether a detail
merge runs on top:

* ``merge="off"`` -- the builder's chunks are the tree, which is what the PDF
  pipeline does today;
* ``merge="real"`` -- the section merge runs against a live endpoint, so the run
  has to be explicitly allowed to call it.

No profile runs the structure-summary stage: a merge is allowed to suggest a
section summary for the runtime tree, but a scored research tree must not contain
text an LLM rewrote.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .tree_builders import BuildResult, build_atomic, build_marker_structure, build_vision_groups

Builder = Callable[[Sequence[Any], str], BuildResult]
"""A tree builder: neutral artifact pages and a document title in, a tree out."""

MergeMode = Literal["off", "real"]


@dataclass(frozen=True)
class Profile:
    """One row of the profile table."""

    name: str
    engine: str
    builder: str
    merge: MergeMode
    structure_summary: bool = False

    def __post_init__(self) -> None:
        if self.builder not in BUILDERS:
            raise ValueError(
                f"profile {self.name!r} names unknown builder {self.builder!r}; "
                f"expected one of {sorted(BUILDERS)}"
            )


BUILDERS: dict[str, Builder] = {
    "vision_groups": build_vision_groups,
    "atomic": build_atomic,
    "marker_structure": build_marker_structure,
}

PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (
        Profile(name="vision_direct", engine="vision", builder="vision_groups", merge="off"),
        Profile(name="marker_llm", engine="marker", builder="marker_structure", merge="real"),
        Profile(name="vision_atomic_merge", engine="vision", builder="atomic", merge="real"),
        Profile(name="vision_atomic", engine="vision", builder="atomic", merge="off"),
        Profile(name="marker_atomic", engine="marker", builder="atomic", merge="off"),
        Profile(name="docling_atomic", engine="docling", builder="atomic", merge="off"),
        Profile(name="native_atomic", engine="native", builder="atomic", merge="off"),
    )
}


def get_profile(name: str) -> Profile:
    """The profile called ``name``, or a ``ValueError`` listing the table."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}") from exc


def builder_for(name: str) -> Builder:
    """The builder a profile runs, so a caller never branches on the name itself."""
    return BUILDERS[get_profile(name).builder]


def build(name: str, pages: Sequence[Any], title: str) -> BuildResult:
    """Build one profile's tree from a neutral artifact."""
    return builder_for(name)(pages, title)
