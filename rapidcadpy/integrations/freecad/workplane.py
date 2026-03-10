"""
FreeCAD Workplane – coordinate-system + fluent CAD API backed by Part module.
"""

import math
from typing import Any, Optional, TYPE_CHECKING

from ...app import App
from ...cad_types import Vector, VectorLike, Vertex
from ...workplane import Workplane
from ...primitives import Line, Circle, Arc

if TYPE_CHECKING:
    from .sketch2d import FreeCADSketch2D
    from .shape import FreeCADShape


class FreeCADWorkplane(Workplane):
    """
    Concrete Workplane backed by FreeCAD's ``Part`` module.

    Inherits the full fluent sketch API (``move_to``, ``line_to``,
    ``rect``, ``circle``, ``three_point_arc``, ``close``, ``extrude``,
    …) from the abstract base class; only FreeCAD-specific behaviour
    is implemented here.
    """

    def __init__(self, app: Optional[Any] = None, *args, **kwargs):
        super().__init__(app=app, *args, **kwargs)
        # Accumulates primitive lists from each .close() call so that a
        # subsequent .extrude() can build a multi-loop (holed) face.
        self._accumulated_loops: list = []
        if hasattr(self.__class__, "normal_vector") or hasattr(self, "normal_vector"):
            self._setup_coordinate_system()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def xy_plane(
        cls, app: Optional[App] = None, offset: Optional[float] = None
    ) -> "FreeCADWorkplane":
        """Workplane on the XY plane (normal = Z)."""
        wp = cls(app=app)
        wp.normal_vector = Vector(0, 0, 1)
        wp._offset = float(offset) if offset is not None else 0.0
        wp._setup_coordinate_system()
        if app is not None:
            app.register_workplane(wp)
        return wp

    @classmethod
    def xz_plane(
        cls, app: Optional[App] = None, offset: Optional[float] = None
    ) -> "FreeCADWorkplane":
        """Workplane on the XZ plane (normal = Y)."""
        wp = cls(app=app)
        wp.normal_vector = Vector(0, 1, 0)
        wp._offset = float(offset) if offset is not None else 0.0
        wp._setup_coordinate_system()
        if app is not None:
            app.register_workplane(wp)
        return wp

    @classmethod
    def yz_plane(
        cls, app: Optional[App] = None, offset: Optional[float] = None
    ) -> "FreeCADWorkplane":
        """Workplane on the YZ plane (normal = X)."""
        wp = cls(app=app)
        wp.normal_vector = Vector(1, 0, 0)
        wp._offset = float(offset) if offset is not None else 0.0
        wp._setup_coordinate_system()
        if app is not None:
            app.register_workplane(wp)
        return wp

    @classmethod
    def from_origin_normal(
        cls,
        app: Optional[App] = None,
        origin: tuple = (0.0, 0.0, 0.0),
        normal: tuple = (0.0, 0.0, 1.0),
    ) -> "FreeCADWorkplane":
        """Create a workplane at an arbitrary world-space origin with the given normal."""
        origin_3d = tuple(origin) if len(origin) == 3 else (origin[0], origin[1], 0.0)  # type: ignore[index]
        normal_3d = tuple(normal) if len(normal) == 3 else (normal[0], normal[1], 0.0)  # type: ignore[index]
        wp = cls(app=app)
        wp.normal_vector = Vector(
            float(normal_3d[0]), float(normal_3d[1]), float(normal_3d[2])
        )
        wp._offset = 0.0
        wp._plane_origin = Vector(
            float(origin_3d[0]), float(origin_3d[1]), float(origin_3d[2])
        )
        wp._setup_coordinate_system()
        if app is not None:
            app.register_workplane(wp)
        return wp

    @classmethod
    def create_offset_plane(
        cls,
        app: App,
        name: str = "XY",
        offset: float = 0.0,
    ) -> "FreeCADWorkplane":
        """Create a named standard plane with an offset along its normal."""
        name_upper = name.upper()
        if name_upper == "XY":
            return cls.xy_plane(app=app, offset=offset)
        elif name_upper == "XZ":
            return cls.xz_plane(app=app, offset=offset)
        elif name_upper == "YZ":
            return cls.yz_plane(app=app, offset=offset)
        else:
            raise ValueError(f"Unknown plane name '{name}'. Use 'XY', 'XZ', or 'YZ'.")

    # ------------------------------------------------------------------
    # State management (required by base-class close() / finish())
    # ------------------------------------------------------------------

    def add(self, loops) -> "FreeCADWorkplane":
        """Accept a loop result for chaining (no-op – pending shapes are already tracked)."""
        return self

    def as_wire(self) -> "FreeCADSketch2D":
        """
        Finalise the current sketch as an open wire (no closing segment).
        Alias for finish() — useful as a pipe spine.
        """
        return self.finish()  # type: ignore[return-value]

    def pipe(self, diameter: float) -> "FreeCADShape":
        """
        Sweep a circular cross-section along the current open sketch path.

        Builds the spine from pending primitives **without** adding a closing
        segment (equivalent to finish().pipe()), so the path stays open.
        """
        from .sketch2d import FreeCADSketch2D

        if not self._pending_shapes:
            raise ValueError("pipe(): no primitives in sketch.")

        sketch = FreeCADSketch2D(
            primitives=list(self._pending_shapes), workplane=self, app=self.app
        )
        self._clear_pending_shapes()
        return sketch.pipe(diameter)  # type: ignore[return-value]

    def _to_3d(self, x: float, y: float) -> tuple:
        """Override to add support for a full 3-D plane origin (_plane_origin)."""
        result = super()._to_3d(x, y)
        origin = getattr(self, "_plane_origin", None)
        if origin is not None:
            return (
                result[0] + float(origin[0]),
                result[1] + float(origin[1]),
                result[2] + float(origin[2]),
            )
        return result

    def _clear_pending_shapes(self) -> None:
        """
        Called by ``close()`` / ``finish()`` after creating a Sketch2D.
        Saves the primitive list into ``_accumulated_loops`` so that a
        later ``extrude()`` call can build a compound multi-loop face.
        Also preserves primitives in ``_extruded_sketches`` for 2-D rendering.
        """
        if self._pending_shapes:
            self._accumulated_loops.append(list(self._pending_shapes))
            self._extruded_sketches.append(list(self._pending_shapes))
        self._pending_shapes = []
        self._current_position = Vertex(0, 0)
        self._loop_start = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 3-D operations
    # ------------------------------------------------------------------

    def extrude(
        self,
        distance: float,
        operation: str = "NewBodyFeatureOperation",
        symmetric: bool = False,
    ) -> "FreeCADShape":
        """
        Extrude the current sketch.

        Handles two cases:
        - **Single open sketch** (primitives still pending): behaves like
          the base-class alias ``close().extrude()``.
        - **Multi-loop sketch** (one or more ``.close()`` calls were made
          first, then ``.extrude()`` is called on the workplane): builds
          a compound ``Part.Face`` from all accumulated loops (outer
          boundary + holes) and extrudes the result.
        """
        import FreeCAD
        import Part
        from .sketch2d import FreeCADSketch2D

        # --- Case 1: still have pending primitives – delegate to base class ---
        if self._pending_shapes:
            # Base-class .extrude() calls close() then sketch.extrude()
            return super().extrude(distance, operation=operation, symmetric=symmetric)  # type: ignore[return-value]

        # --- Case 2: multi-loop sketch accumulated via close() calls ----------
        if not self._accumulated_loops:
            raise ValueError("No primitives to extrude.")

        loops = self._accumulated_loops
        self._accumulated_loops = []

        # Build one wire per loop
        wires = []
        for primitive_list in loops:
            sketch = FreeCADSketch2D(
                primitives=primitive_list, workplane=self, app=self.app
            )
            wire = sketch._make_wire()
            if wire is not None:
                wires.append(wire)

        if not wires:
            raise ValueError("No valid wires could be built from accumulated loops.")

        # Sort wires by descending bounding-box area so the outer loop comes first.
        def _wire_area(w):
            bb = w.BoundBox
            return bb.XLength * bb.YLength

        wires.sort(key=_wire_area, reverse=True)

        # Build compound face: Part.Face([outer, hole1, hole2, ...])
        try:
            face = Part.Face(wires)
            if not face.isValid():
                raise ValueError("Compound face is not valid.")
        except Exception as exc:
            raise ValueError(f"Multi-loop face construction failed: {exc}") from exc

        n = self.normal_vector
        if symmetric:
            half = distance / 2.0
            face.translate(
                FreeCAD.Vector(
                    float(n[0]) * (-half),
                    float(n[1]) * (-half),
                    float(n[2]) * (-half),
                )
            )

        extrude_vec = FreeCAD.Vector(
            float(n[0]) * distance,
            float(n[1]) * distance,
            float(n[2]) * distance,
        )

        solid = face.extrude(extrude_vec)

        # Use the sketch helper to apply the operation
        dummy_sketch = FreeCADSketch2D(primitives=[], workplane=self, app=self.app)
        return dummy_sketch._apply_operation(solid, operation)  # type: ignore[return-value]

    def box(
        self,
        length: float,
        width: float,
        height: float,
        centered: bool = True,
    ) -> "FreeCADShape":
        """
        Create a box aligned with the workplane coordinate system.

        Args:
            length:   Dimension along the workplane X axis.
            width:    Dimension along the workplane Y axis.
            height:   Dimension along the workplane normal (Z) axis.
            centered: Centre the box at the current 2-D cursor position.

        Returns:
            FreeCADShape wrapping the solid.
        """
        import FreeCAD
        import Part
        from .shape import FreeCADShape

        if not hasattr(self, "_local_x"):
            self._setup_coordinate_system()

        # Local corner when centred
        if centered:
            lx = -length / 2.0
            ly = -width / 2.0
            lz = -height / 2.0
        else:
            lx, ly, lz = 0.0, 0.0, 0.0

        # Build box at local origin
        box_shape = Part.makeBox(length, width, height, FreeCAD.Vector(lx, ly, lz))

        # Transform from local to world coordinates using the workplane axes
        center_2d = self._current_position
        cx, cy, cz = self._to_3d(center_2d.x, center_2d.y)

        lx_v = self._local_x
        ly_v = self._local_y
        lz_v = self._local_z

        m = FreeCAD.Matrix()
        # Columns are the local x, y, z axes expressed in world space
        m.A11, m.A12, m.A13, m.A14 = lx_v.x, ly_v.x, lz_v.x, float(cx)
        m.A21, m.A22, m.A23, m.A24 = lx_v.y, ly_v.y, lz_v.y, float(cy)
        m.A31, m.A32, m.A33, m.A34 = lx_v.z, ly_v.z, lz_v.z, float(cz)
        m.A41, m.A42, m.A43, m.A44 = 0.0, 0.0, 0.0, 1.0

        box_shape = box_shape.transformGeometry(m)
        return FreeCADShape(box_shape, self.app)

    def revolve(
        self,
        angle: float,
        axis: str = "Z",
        operation: str = "NewBodyFeatureOperation",
    ) -> "FreeCADShape":
        """
        Revolve the current sketch around a world axis.

        Args:
            angle:     Revolution angle **in radians**.
            axis:      World axis to revolve around – "X", "Y", or "Z".
            operation: Same semantics as ``Sketch2D.extrude``.

        Returns:
            FreeCADShape wrapping the resulting solid.
        """
        import FreeCAD
        from .sketch2d import FreeCADSketch2D

        if not self._pending_shapes:
            if getattr(self.app, "silent_geometry_failures", False):
                return None  # type: ignore[return-value]
            raise ValueError("No shapes to revolve – sketch is empty.")

        sketch2d = FreeCADSketch2D(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )
        face = sketch2d._make_face()
        if face is None:
            return None  # type: ignore[return-value]

        axis_map = {
            "X": FreeCAD.Vector(1, 0, 0),
            "Y": FreeCAD.Vector(0, 1, 0),
            "Z": FreeCAD.Vector(0, 0, 1),
        }
        revolve_dir = axis_map.get(axis.upper(), FreeCAD.Vector(0, 0, 1))
        angle_deg = math.degrees(angle)

        try:
            solid = face.revolve(FreeCAD.Vector(0, 0, 0), revolve_dir, angle_deg)
        except Exception as exc:
            if getattr(self.app, "silent_geometry_failures", False):
                return None  # type: ignore[return-value]
            raise RuntimeError(f"Revolve failed: {exc}") from exc

        self._clear_pending_shapes()
        return sketch2d._apply_operation(solid, operation)

    def sweep(
        self,
        profile: "FreeCADWorkplane",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> "FreeCADShape":
        """
        Sweep *profile*'s sketch along this workplane's sketch path.

        Args:
            profile:         Cross-section workplane.
            make_solid:      Produce a solid or a shell.
            is_frenet:       Keep profile perpendicular to the path (Frenet frame).
            transition_mode: Not used (kept for API parity with OCC backend).

        Returns:
            FreeCADShape wrapping the resulting solid.
        """
        from .sketch2d import FreeCADSketch2D

        if not self._pending_shapes:
            raise ValueError("No shapes in path sketch.")
        if not profile._pending_shapes:
            raise ValueError("No shapes in profile sketch.")

        path_sketch = FreeCADSketch2D(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )
        profile_sketch = FreeCADSketch2D(
            primitives=profile._pending_shapes, workplane=profile, app=self.app
        )

        result = path_sketch.sweep(
            profile=profile_sketch,
            make_solid=make_solid,
            is_frenet=is_frenet,
            transition_mode=transition_mode,
        )

        self._clear_pending_shapes()
        return result  # type: ignore[return-value]
