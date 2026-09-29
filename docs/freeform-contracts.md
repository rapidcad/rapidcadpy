# Freeform modeling contracts and live integrations

Shared intent and live factory extension points are implemented. FreeCAD also
supports persistent profiles captured from fluent line, circle and arc geometry,
profile updates, planar spline profiles/paths, native lofts and linked native
sweeps. Spatial spline paths remain `unsupported`. Existing fluent APIs remain
available.

## Modeling intent

`rapidcadpy.modeling` contains immutable dataclasses with JSON-compatible
`to_dict()` payloads. `modeling_request_from_dict()` reconstructs requests and
validates their inputs. The `kind` discriminator distinguishes requests and
curve representations. Unknown fields are rejected. There is no Pydantic or
CAD-kernel dependency in this module; agent schemas should map into these types.

| Contract | Meaning |
| --- | --- |
| `InterpolatedSpline` | Pass through ordered points within tolerance, using the requested degree. |
| `ControlPointSpline` | Construct a clamped B-spline/NURBS from poles, optional weights and an optional expanded knot vector. Poles are not interpolation points. |
| `CoordinateFrame` | World-space origin, unit X axis and perpendicular unit normal; local Y is normal cross X. |
| `ProfileDefinition` | One ordered planar boundary in a coordinate frame, with a requested open/closed state. |
| `PathDefinition` | Ordered planar or spatial spline segments in a coordinate frame. |
| `CadProfile`, `CadPath` | Persistent document/object IDs, not workplanes or native handles. |
| `LoftDefinition` | Ordered, distinct profile references from one document; solid/shell, ruled/smooth and orientation/seam alignment intent. |
| `SweepDefinition` | Profile and path references from one document, solid/shell intent, orientation and corner transition. |

Coordinates and tolerances use the containing frame's `unit`: `mm` (default),
`cm`, `m`, or `in`. Frame origins use the same unit. Axes, knots and weights are
dimensionless. Profile geometry lies in local XY; path geometry can use local Z.
Adapters must convert units at the kernel boundary, including tolerances.

Spline degree is exact intent, not permission to silently select another degree.
Omitted weights mean one per pole. Omitted knots mean a uniform clamped vector:
for `n` poles and degree `p`, repeat 0 and 1 each `p + 1` times and place
`n - p - 1` evenly spaced interior knots between them. Explicit knots are
expanded, nondecreasing, and have length `n + p + 1`. End multiplicities must be
`p + 1`; interior multiplicity cannot exceed `p`.

Closure means positional (C0) closure only. The native adapter closes interpolation
by appending the starting point, without requesting periodicity. Closed interpolation omits the
repeated endpoint. Closed control-point curves repeat the first pole at the end
within tolerance. Periodic curves and prescribed tangent/curvature continuity
are outside this first contract. Profile connectivity, self-intersection, actual
closure and loft/sweep feasibility require native validation. Multiple boundary
loops and holes remain future extensions. Typed line, arc and circle segments can
now mix with splines in one ordered boundary.

Sweep `orientation` is `frenet` or `corrected_frenet`; `transition` is
`transformed`, `right`, or `round`. These express requested behavior, not a
promise that every backend supports every combination. Adapters inspect and
reject unsupported modes rather than ignoring them. Profiles retain their
specified placement; this contract does not request automatic profile alignment.

```python
from rapidcadpy import (
    CoordinateFrame, InterpolatedSpline, ProfileDefinition,
    CadProfile, LoftDefinition, modeling_request_from_dict,
)

profile = ProfileDefinition(
    curves=(InterpolatedSpline(
        points=((0, 0, 0), (20, 5, 0), (20, 20, 0), (0, 20, 0)),
        degree=3,
        closed=True,
    ),),
    frame=CoordinateFrame(origin=(0, 0, 10), unit="mm"),
)
assert modeling_request_from_dict(profile.to_dict()) == profile

# IDs are returned by a supported create-profile operation, not invented
# by the caller. Merely constructing a definition never creates CAD geometry.
loft = LoftDefinition(profiles=(
    CadProfile(document_id="doc-id", object_id="profile-id-1"),
    CadProfile(document_id="doc-id", object_id="profile-id-2"),
))
```

## Live abstract factory

`LiveCadBackendRegistry` is injected into `CadSession`. Its lazy FreeCAD default
does not load the FreeCAD kernel. Additional integrations register a factory
constructor on that registry instance:

```python
from rapidcadpy import CadSession, LiveCadBackendRegistry

registry = LiveCadBackendRegistry()
registry.register("another_cad", AnotherCadLiveBackendFactory)
session = CadSession(execution_mode="gui", live_backend_registry=registry)
session.list_cad_applications()
session.select_cad_application("another_cad", target_id="running-instance-id")
```

A `LiveCadBackendFactory` supplies discovery and connection plus three products
bound to that connection: `ModelingBackend`, `DocumentHydrator`, and
`CapabilityInspector`. `LiveCadBackend` groups those compatible services.
Connection `close()` releases transport resources without closing the user's CAD
application. Failed factory construction releases its candidate connection.
Failed selection leaves the previously selected service family active.

Generic session calls use the selected connection. Existing FreeCAD-specific
attachment/launch entry points remain compatibility paths. Headless setup and
the application's isolated artifact `CadBackend` factory are unchanged; registering
a live integration does not automatically implement either of those paths.

`session.inspect_modeling_support(definition)` reports request-level support.
`session.apply_modeling(definition, expected_revision=revision)` dispatches shared
intent to the selected modeling product. The factory's legacy capability list
describes existing session methods; request inspection is authoritative for the
new persistent-profile contracts. FreeCAD accepts all four definitions within the
planar restrictions described below. Request inspection stays kernel-free and
reports unsupported degree/frame/alignment intent before contacting the application.

Native modeling implementations must validate the expected revision, execute
atomically, recompute and validate the feature tree, rehydrate runtime state, and
return `CadOperationResult` with revision, changed/created/removed IDs, support
and warnings. Failed mutations must restore native and runtime state. Profile
creation returns `data["profile"]` as a serialized `CadProfile`; path creation
returns `data["path"]` as a serialized `CadPath`. Loft/sweep creation returns
`data["object_id"]` for the resulting feature. References must retain their
identity across hydration and native save/reopen; input profiles are dependencies
and must not be consumed or silently copied into disconnected geometry.

## Persistent FreeCAD profiles and transactional lofts

A workplane holds a coordinate frame and pending fluent geometry. A persistent
profile is a native `Sketcher::SketchObject` with its own stable ID. Capturing a
profile does not clear pending geometry or consume a closed loop. Repeated lofts
from unchanged workplanes reuse the same sketches. Native `Part::Loft.Sections`
links directly to those sketches, preserving editable dependencies.

```python
session = CadSession(execution_mode="embedded")
session.new_document("Profiles")
a = session.work_plane("XY")["object_id"]
session.circle(10)
b = session.work_plane("XY", offset=30)["object_id"]
session.circle(15)
first = session.create_profile(a, expected_revision=session.document_revision)
second = session.create_profile(b, expected_revision=first["document_revision"])
result = session.loft(
    profile_ids=[first["object_id"], second["object_id"]],
    make_solid=True,
    expected_revision=second["document_revision"],
)
```

`create_path(workplane_id, expected_revision=...)` captures an open or closed
sketch with a distinct `CadPath` reference. Both capture operations currently
accept one boundary made of lines, circles or arcs. Multiple loops are rejected.
`update_profile(profile_id, workplane_id, expected_revision=...)` replaces geometry
in the same native sketch and recomputes dependent lofts. Constrained sketches
are rejected rather than losing their constraints. Per-object operation support
reports this restriction. Persistent-profile updates invalidate capture caches.

Legacy `session.loft([workplane_id, ...])` captures/reuses profiles automatically;
`workplane.loft(other_workplane)` uses the same native lifecycle when attached to
an application. Standalone geometry retains its existing fluent behavior.
`make_solid=True` requires closed sections and exactly one resulting solid;
`False` requires a shell without solids. Results report `object_type` accordingly.

The shared mutation coordinator checks native state against `expected_revision`,
validates inputs, opens a transaction, changes native features, recomputes and
validates the tree, rehydrates dependencies, serializes the result, then commits.
Any failure restores Python-owned references, caches, pending geometry and
selection alongside the native document. FreeCAD metadata is restored explicitly
because native undo does not include it. An existing user transaction is rejected
without being modified. Revision mismatches return the current native revision;
callers can refresh before retrying. Convenience wrappers permit omitted revisions;
`apply_modeling` requires a nonempty revision.

Profile/path/loft IDs are stored as native properties; the document ID is stored
in document metadata. These survive FCStd save/reopen. Revisions are stable across
unchanged hydration but may change on reopen because FreeCAD normalizes native
B-Reps and attachment state; use the revision returned by hydration after opening.
No additional MCP server or Pydantic tool is introduced here.


## Native spline geometry

`apply_modeling(ProfileDefinition(...), expected_revision=...)` and
`apply_modeling(PathDefinition(...), expected_revision=...)` create persistent
`Sketcher::SketchObject` geometry. Both interpolation and control-point inputs
are supported. FreeCAD interpolation supports degree 3 only; another requested
degree is rejected. Clamped control-point splines support degrees 1–25, weights
and expanded knots; omitted knots follow the shared contract. Lengths, origin
and tolerance are converted to millimeters; native placement retains the frame.

Segments must connect in their given order. A profile must match its requested
closure and, when closed, form a valid face. Paths must lie in local XY even when
the sketch frame is tilted in world space. Spatial `PathDefinition` inputs are
rejected explicitly, pending a persistent recomputable 3D representation.
`update_path(path_id, definition, expected_revision=...)` replaces geometry in the
same unconstrained sketch and recomputes its native dependents atomically.

## Loft orientation and seams

`LoftDefinition.alignment` and `session.loft(..., alignment=...)` default to
`automatic`. Sections remain in exactly the caller's order and link to the
original persistent sketches. FreeCAD's native loft invokes kernel compatibility
matching for edge correspondence, winding and wire origin. It does not expose a
switch to disable this, so `alignment="preserve"` is explicitly unsupported.
See [FreeCAD's loft builder](https://github.com/FreeCAD/FreeCAD/blob/main/src/Mod/Part/App/TopoShapeExpansion.cpp).

Automatic matching cannot reliably relocate a seam within a single closed
B-spline edge. For closed single-edge sections, this adapter therefore requires
parallel, consistently oriented frames, matching winding, and aligned directions
from each face centroid to its seam/start point. Callers must supply corresponding
start points and traversal directions. Ambiguous inputs fail before creating the
loft rather than modifying profiles to compensate. The same checks run after
section edits, so an incompatible update rolls back. This is a conservative
supported subset, not a guarantee for every possible freeform loft.

## Native planar sweeps

`session.sweep(profile_id, path_id, ..., expected_revision=...)` and
`apply_modeling(SweepDefinition(...), expected_revision=...)` create a
`Part::Sweep` with `Sections=[profile_sketch]` and `Spine=(path_sketch, [])`.
The whole spine is linked, avoiding fragile numbered edge selections. The first
increment supports one closed profile and one open planar sketch path, with a
solid or shell result. Profiles and paths remain reusable dependencies.

| Shared option | Native setting |
| --- | --- |
| `orientation="frenet"` | `Frenet=True` |
| `orientation="corrected_frenet"` | `Frenet=False` |
| `transition="transformed"` | `Transition="Transformed"` |
| `transition="right"` | `Transition="Right corner"` |
| `transition="round"` | `Transition="Round corner"` |

These map to [FreeCAD's native sweep settings](https://github.com/FreeCAD/FreeCAD/blob/main/src/Mod/Part/App/PartFeatures.cpp).
Unknown frame/corner modes are rejected, never substituted. Kernel feasibility
remains conditional on the actual path and section; a failed corner or surface
construction restores the document and runtime mirror.

The section plane must pass through the spine start and its normal must be
parallel to the initial tangent. Automatic section translation/rotation is not
supported. Editing either dependency must preserve these requirements; invalid
updates roll back. Native validation also requires a valid result and exactly
one solid with positive volume, or a shell without solids. References, settings
and dependency links survive hydration and FCStd save/reopen.

```python
from rapidcadpy import (
    CadPath, CadProfile, ControlPointSpline, PathDefinition, SweepDefinition,
)

created = session.new_document("Sweep")
path = session.apply_modeling(
    PathDefinition(curves=(ControlPointSpline(
        control_points=((0, 0, 0), (0, 10, 0), (10, 20, 0), (10, 30, 0)),
    ),)),
    expected_revision=created["document_revision"],
)
workplane = session.work_plane("XZ")["object_id"]
session.circle(1)
profile = session.create_profile(workplane, expected_revision=path["document_revision"])
sweep = session.apply_modeling(
    SweepDefinition(
        profile=CadProfile(**profile["profile"]),
        path=CadPath(**path["path"]),
        orientation="corrected_frenet", transition="round",
    ),
    expected_revision=profile["document_revision"],
)
```


## Harness and editable feature contracts

`LineSegment`, `ArcSegment` and `CircleSegment` extend the shared curve boundary
alongside spline inputs. The Pydantic harness maps its `segments` field into these
RapidCADPy definitions; validation remains in the neutral library. Public session
`create_profile`/`create_path` accept a `definition` in addition to their legacy
workplane convenience path. `update_profile` can replace geometry from a typed
`ProfileDefinition` while retaining the original native sketch and dependencies.

`LoftFeatureUpdate` and `SweepFeatureUpdate` describe a bounded parameter patch;
`feature_update_from_dict` rejects undeclared fields. `update_feature` modifies the
same native object inside the shared mutation lifecycle. At least one parameter
is required. Geometry validity, loft correspondence, sweep placement and document
revision checks apply to edits as well as creation.

`inspect_capabilities` reports operation support and restrictions; `inspect_geometry`
reports refreshed validity, dimensions in millimeters, dependencies, dependents,
pending recomputation, and native failure details. The Pydantic `inspect_design`
tool attaches this report alongside its existing image. Native handles stay in
RapidCADPy and native property names are absent from harness arguments.

The GUI connector advertises and negotiates `rapidcad.live` major 1/minor 0 with
an explicit method list. Both transports carry the agreement on each versioned
command. Old/missing/incompatible manifests remove unavailable capabilities and
block those calls with `cad_bridge_incompatible`. A cached older session class
cannot acquire new capabilities just by reloading the bridge module. Existing
connector installations must be updated and restarted before new tools are used.
