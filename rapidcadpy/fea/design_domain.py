from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DesignDomain:
    """
    Represents a design domain geometry that can be non-cubic.

    Supports:
    - Simple primitives: box, cylinder, sphere
    - Composite shapes: L-shape, T-shape, plus
    - CSG operations: union, subtract, intersect
    """

    units: str = "mm"
    shape_type: str = "box"  # box, cylinder, sphere, l_shape, t_shape, csg
    bounds: Optional[Dict[str, float]] = None  # For box: x_min, x_max, etc.
    params: Optional[Dict[str, Any]] = None  # Shape-specific parameters
    operations: Optional[List[Dict[str, Any]]] = None  # CSG operations

    def describe(self) -> str:
        """
        Generate a natural language description of the design domain for the LLM.

        This description helps the planner understand the valid spatial region where
        geometry can be created.
        """
        desc = ["Design Domain Configuration:"]
        desc.append(f"- Shape Type: {self.shape_type}")
        desc.append(f"- Units: {self.units}")

        if self.shape_type == "box" and self.bounds:
            desc.append(f"- Bounding Box: {self.bounds}")
            desc.append("  (Rectangular design space)")

        elif self.shape_type in ["l_shape", "t_shape", "plus"]:
            desc.append(f"- Parameters: {self.params}")
            desc.append(f"  (Composite {self.shape_type} design space)")

        elif self.shape_type == "cylinder":
            desc.append(f"- Cylinder Parameters: {self.params}")

        elif self.shape_type == "sphere":
            desc.append(f"- Sphere Parameters: {self.params}")

        elif self.shape_type == "csg" and self.operations:
            desc.append(
                f"- Constructive Solid Geometry (CSG) with {len(self.operations)} operations:"
            )
            for i, op in enumerate(self.operations):
                op_type = op.get("type", "unknown")
                operation = op.get("operation", "add")
                params = op.get("params", {})
                desc.append(f"  {i+1}. {operation.upper()} {op_type} with {params}")

        # Always calculate and add the bounding box as a reference
        try:
            bb = self.get_bounding_box()
            desc.append(f"- Overall Bounding Box: {bb}")
        except Exception as e:
            logger.warning(f"Could not calculate bounding box for description: {e}")

        return "\n".join(desc)

    def build_geometry(self, app=None) -> "Shape":
        """
        Build the design domain geometry using rapidcadpy.

        Returns:
            rapidcadpy Shape with the design domain solid
        """
        if app is None:
            from rapidcadpy.integrations.ocp.app import OpenCascadeOcpApp
            app = OpenCascadeOcpApp()

        if self.shape_type == "box":
            return self._build_box(app)
        elif self.shape_type == "cylinder":
            return self._build_cylinder(app)
        elif self.shape_type == "sphere":
            return self._build_sphere(app)
        elif self.shape_type == "l_shape":
            return self._build_l_shape(app)
        elif self.shape_type == "t_shape":
            return self._build_t_shape(app)
        elif self.shape_type == "plus":
            return self._build_plus(app)
        elif self.shape_type == "csg":
            return self._build_csg(app)
        else:
            return self._build_box(app)

    def _build_box(self, app) -> "Shape":
        if not self.bounds:
            raise ValueError("Box shape requires 'bounds' with x_min, x_max, etc.")
        x_min = self.bounds.get("x_min", 0)
        x_max = self.bounds.get("x_max", 100)
        y_min = self.bounds.get("y_min", 0)
        y_max = self.bounds.get("y_max", 100)
        z_min = self.bounds.get("z_min", 0)
        z_max = self.bounds.get("z_max", 100)
        wp = app.work_plane("XY", offset=z_min)
        wp.move_to(x_min, y_min).rect(x_max - x_min, y_max - y_min, centered=False)
        return wp.extrude(z_max - z_min)

    def _build_cylinder(self, app) -> "Shape":
        p = self.params or {}
        radius = p.get("radius", 50)
        height = p.get("height", 100)
        cx, cy, cz = p.get("center", [0, 0, 0])
        axis = p.get("axis", "z").lower()
        if axis == "x":
            wp = app.work_plane("YZ", offset=cx)
            wp.move_to(cy, cz).circle(radius)
        elif axis == "y":
            wp = app.work_plane("XZ", offset=cy)
            wp.move_to(cx, cz).circle(radius)
        else:
            wp = app.work_plane("XY", offset=cz)
            wp.move_to(cx, cy).circle(radius)
        return wp.extrude(height)

    def _build_sphere(self, app) -> "Shape":
        p = self.params or {}
        radius = p.get("radius", 50)
        cx, cy, cz = p.get("center", [0, 0, 0])
        wp = app.work_plane("XY", offset=cz)
        wp.move_to(cx, cy - radius).line_to(cx, cy + radius)
        wp.three_point_arc((cx + radius, cy), (cx, cy - radius))
        return wp.close().revolve(360, axis="Y")

    def _build_l_shape(self, app) -> "Shape":
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        leg_width = p.get("leg_width", width * 0.4)
        base_height = p.get("base_height", height * 0.3)
        wp = app.work_plane("XY")
        wp.move_to(0, 0).rect(width, depth, centered=False)
        base = wp.extrude(base_height)
        wp2 = app.work_plane("XY", offset=base_height)
        wp2.move_to(0, 0).rect(leg_width, depth, centered=False)
        return base.union(wp2.extrude(height - base_height))

    def _build_t_shape(self, app) -> "Shape":
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        stem_w = p.get("stem_width", width * 0.4)
        cap_h = p.get("cap_height", height * 0.3)
        stem_h = height - cap_h
        stem_x = (width - stem_w) / 2
        wp = app.work_plane("XY")
        wp.move_to(stem_x, 0).rect(stem_w, depth, centered=False)
        stem = wp.extrude(stem_h)
        wp2 = app.work_plane("XY", offset=stem_h)
        wp2.move_to(0, 0).rect(width, depth, centered=False)
        return stem.union(wp2.extrude(cap_h))

    def _build_plus(self, app) -> "Shape":
        p = self.params or {}
        w = p.get("width", 200)
        h = p.get("height", 200)
        d = p.get("depth", 100)
        arm_w = p.get("arm_width", w * 0.4)
        wp = app.work_plane("XY", offset=(h - arm_w) / 2)
        wp.move_to(0, 0).rect(w, d, centered=False)
        h_bar = wp.extrude(arm_w)
        wp2 = app.work_plane("XY")
        wp2.move_to((w - arm_w) / 2, 0).rect(arm_w, d, centered=False)
        return h_bar.union(wp2.extrude(h))

    def _build_csg(self, app) -> "Shape":
        result = None
        for op in self.operations:
            op_type = op.get("type", "box")
            oper = op.get("operation", "add")
            params = op.get("params", {})
            shape = None
            if op_type == "box":
                wp = app.work_plane("XY", offset=params.get("z", 0))
                wp.move_to(params.get("x", 0), params.get("y", 0)).rect(params.get("dx", 100), params.get("dy", 100), centered=False)
                shape = wp.extrude(params.get("dz", 100))
            elif op_type == "cylinder":
                cx, cy, cz = params.get("cx", 0), params.get("cy", 0), params.get("cz", 0)
                axis = params.get("axis", "z").lower()
                r, ht = params.get("radius", 50), params.get("height", 100)
                if axis == "x":
                    wp = app.work_plane("YZ", offset=cx)
                    wp.move_to(cy, cz).circle(r)
                elif axis == "y":
                    wp = app.work_plane("XZ", offset=cy)
                    wp.move_to(cx, cz).circle(r)
                else:
                    wp = app.work_plane("XY", offset=cz)
                    wp.move_to(cx, cy).circle(r)
                shape = wp.extrude(ht)
            elif op_type == "sphere":
                cx, cy, cz = params.get("cx", 0), params.get("cy", 0), params.get("cz", 0)
                r = params.get("radius", 50)
                wp = app.work_plane("XY", offset=cz)
                wp.move_to(cx, cy - r).line_to(cx, cy + r).three_point_arc((cx + r, cy), (cx, cy - r))
                shape = wp.close().revolve(360, axis="Y")
            if shape:
                if result is None or oper == "add":
                    result = shape if result is None else result.union(shape)
                elif oper == "subtract":
                    result = result.cut(shape)
                elif oper == "intersect":
                    if hasattr(result, "intersect"): result = result.intersect(shape)
        if result is None: raise ValueError("No geometry")
        return result

    def export_step(self, filepath: str) -> str:
        """Export the design domain to a STEP file."""
        geometry = self.build_geometry()
        geometry.to_step(filepath)
        logger.info(f"Exported design domain to {filepath}")
        return filepath

    def get_bounding_box(self) -> Dict[str, float]:
        """
        Get the axis-aligned bounding box of the design domain.

        For well-known primitive shapes (box, cylinder, sphere, l_shape, t_shape,
        plus) the bounding box is computed analytically without building the full
        CadQuery geometry.  CSG and unknown shapes fall back to the geometry build.

        Returns:
            Dict with x_min, x_max, y_min, y_max, z_min, z_max
        """
        if self.shape_type == "box" and self.bounds:
            return self.bounds

        p = self.params or {}

        if self.shape_type == "cylinder":
            radius = float(p.get("radius", 50))
            height = float(p.get("height", 100))
            center = list(p.get("center", [0, 0, 0]))
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            axis = str(p.get("axis", "z")).lower()

            # The cylinder base is placed at the origin (not-centered in Z),
            # then translated by `center`.  Axis rotations move the extrusion
            # direction from Z to the target axis.
            if axis == "z":
                return {
                    "x_min": cx - radius,
                    "x_max": cx + radius,
                    "y_min": cy - radius,
                    "y_max": cy + radius,
                    "z_min": cz,
                    "z_max": cz + height,
                }
            elif axis == "x":
                return {
                    "x_min": cx,
                    "x_max": cx + height,
                    "y_min": cy - radius,
                    "y_max": cy + radius,
                    "z_min": cz - radius,
                    "z_max": cz + radius,
                }
            elif axis == "y":
                return {
                    "x_min": cx - radius,
                    "x_max": cx + radius,
                    "y_min": cy,
                    "y_max": cy + height,
                    "z_min": cz - radius,
                    "z_max": cz + radius,
                }

        elif self.shape_type == "sphere":
            radius = float(p.get("radius", 50))
            center = list(p.get("center", [0, 0, 0]))
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            return {
                "x_min": cx - radius,
                "x_max": cx + radius,
                "y_min": cy - radius,
                "y_max": cy + radius,
                "z_min": cz - radius,
                "z_max": cz + radius,
            }

        elif self.shape_type == "l_shape":
            width = float(p.get("width", 200))
            height = float(p.get("height", 200))
            depth = float(p.get("depth", 100))
            # Union of base (full width) and vertical leg → overall box
            return {
                "x_min": 0.0,
                "x_max": width,
                "y_min": 0.0,
                "y_max": depth,
                "z_min": 0.0,
                "z_max": height,
            }

        elif self.shape_type == "t_shape":
            width = float(p.get("width", 200))
            height = float(p.get("height", 200))
            depth = float(p.get("depth", 100))
            return {
                "x_min": 0.0,
                "x_max": width,
                "y_min": 0.0,
                "y_max": depth,
                "z_min": 0.0,
                "z_max": height,
            }

        elif self.shape_type == "plus":
            width = float(p.get("width", 200))
            height = float(p.get("height", 200))
            depth = float(p.get("depth", 100))
            return {
                "x_min": 0.0,
                "x_max": width,
                "y_min": 0.0,
                "y_max": depth,
                "z_min": 0.0,
                "z_max": height,
            }

        # For CSG and unknown shapes fall back to building the geometry
        geometry = self.build_geometry()
        
        try:
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib
            bnd_box = Bnd_Box()
            BRepBndLib.Add_s(geometry.obj, bnd_box)
            xmin, ymin, zmin, xmax, ymax, zmax = bnd_box.Get()
            return {
                "x_min": xmin, "x_max": xmax,
                "y_min": ymin, "y_max": ymax,
                "z_min": zmin, "z_max": zmax
            }
        except Exception:
            return {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0, "z_min": 0, "z_max": 0}
