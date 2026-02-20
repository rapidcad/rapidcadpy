from dataclasses import dataclass
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

    def build_geometry(self) -> "cq.Workplane":
        """
        Build the design domain geometry using CadQuery.

        Returns:
            CadQuery Workplane with the design domain solid
        """
        import cadquery as cq

        if self.shape_type == "box":
            return self._build_box(cq)
        elif self.shape_type == "cylinder":
            return self._build_cylinder(cq)
        elif self.shape_type == "sphere":
            return self._build_sphere(cq)
        elif self.shape_type == "l_shape":
            return self._build_l_shape(cq)
        elif self.shape_type == "t_shape":
            return self._build_t_shape(cq)
        elif self.shape_type == "plus":
            return self._build_plus(cq)
        elif self.shape_type == "csg":
            return self._build_csg(cq)
        else:
            logger.warning(
                f"Unknown shape type '{self.shape_type}', falling back to box"
            )
            return self._build_box(cq)

    def _build_box(self, cq) -> "cq.Workplane":
        """Build a box from bounds."""
        if not self.bounds:
            raise ValueError("Box shape requires 'bounds' with x_min, x_max, etc.")

        x_min = self.bounds.get("x_min", 0)
        x_max = self.bounds.get("x_max", 100)
        y_min = self.bounds.get("y_min", 0)
        y_max = self.bounds.get("y_max", 100)
        z_min = self.bounds.get("z_min", 0)
        z_max = self.bounds.get("z_max", 100)

        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min

        return (
            cq.Workplane("XY")
            .box(dx, dy, dz, centered=False)
            .translate((x_min, y_min, z_min))
        )

    def _build_cylinder(self, cq) -> "cq.Workplane":
        """
        Build a cylinder from params.

        Params:
            radius: Cylinder radius
            height: Cylinder height
            center: (x, y, z) center of base
            axis: 'x', 'y', or 'z' (default 'z')
        """
        p = self.params or {}
        radius = p.get("radius", 50)
        height = p.get("height", 100)
        center = p.get("center", [0, 0, 0])
        axis = p.get("axis", "z").lower()

        # Build cylinder along Z, then rotate if needed
        cyl = cq.Workplane("XY").cylinder(height, radius, centered=(True, True, False))

        if axis == "x":
            cyl = cyl.rotate((0, 0, 0), (0, 1, 0), 90)
        elif axis == "y":
            cyl = cyl.rotate((0, 0, 0), (1, 0, 0), -90)
        # axis == 'z' is default, no rotation needed

        return cyl.translate(tuple(center))

    def _build_sphere(self, cq) -> "cq.Workplane":
        """
        Build a sphere from params.

        Params:
            radius: Sphere radius
            center: (x, y, z) center
        """
        p = self.params or {}
        radius = p.get("radius", 50)
        center = p.get("center", [0, 0, 0])

        return cq.Workplane("XY").sphere(radius).translate(tuple(center))

    def _build_l_shape(self, cq) -> "cq.Workplane":
        """
        Build an L-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            leg_width: Width of the vertical leg
            base_height: Height of the horizontal base
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        leg_width = p.get("leg_width", width * 0.4)
        base_height = p.get("base_height", height * 0.3)

        # Horizontal base
        base = cq.Workplane("XY").box(width, depth, base_height, centered=False)

        # Vertical leg on the left
        leg = cq.Workplane("XY").box(leg_width, depth, height, centered=False)

        return base.union(leg)

    def _build_t_shape(self, cq) -> "cq.Workplane":
        """
        Build a T-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            stem_width: Width of the vertical stem
            cap_height: Height of the horizontal cap
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        stem_width = p.get("stem_width", width * 0.4)
        cap_height = p.get("cap_height", height * 0.3)

        stem_height = height - cap_height
        stem_x = (width - stem_width) / 2

        # Vertical stem
        stem = (
            cq.Workplane("XY")
            .box(stem_width, depth, stem_height, centered=False)
            .translate((stem_x, 0, 0))
        )

        # Horizontal cap at top
        cap = (
            cq.Workplane("XY")
            .box(width, depth, cap_height, centered=False)
            .translate((0, 0, stem_height))
        )

        return stem.union(cap)

    def _build_plus(self, cq) -> "cq.Workplane":
        """
        Build a plus/cross-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            arm_width: Width of each arm
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        arm_width = p.get("arm_width", width * 0.4)

        # Horizontal bar
        h_bar = (
            cq.Workplane("XY")
            .box(width, depth, arm_width, centered=False)
            .translate((0, 0, (height - arm_width) / 2))
        )

        # Vertical bar
        v_bar = (
            cq.Workplane("XY")
            .box(arm_width, depth, height, centered=False)
            .translate(((width - arm_width) / 2, 0, 0))
        )

        return h_bar.union(v_bar)

    def _build_csg(self, cq) -> "cq.Workplane":
        """
        Build geometry using CSG operations.

        Operations is a list of dicts with:
            - type: 'box', 'cylinder', 'sphere'
            - operation: 'add' (union), 'subtract', 'intersect'
            - params: shape-specific parameters
        """
        if not self.operations:
            raise ValueError("CSG shape requires 'operations' list")

        result = None

        for i, op in enumerate(self.operations):
            op_type = op.get("type", "box")
            operation = op.get("operation", "add")
            params = op.get("params", {})

            # Build the primitive
            if op_type == "box":
                x = params.get("x", 0)
                y = params.get("y", 0)
                z = params.get("z", 0)
                dx = params.get("dx", 100)
                dy = params.get("dy", 100)
                dz = params.get("dz", 100)
                shape = (
                    cq.Workplane("XY")
                    .box(dx, dy, dz, centered=False)
                    .translate((x, y, z))
                )
            elif op_type == "cylinder":
                cx = params.get("cx", 0)
                cy = params.get("cy", 0)
                cz = params.get("cz", 0)
                radius = params.get("radius", 50)
                height = params.get("height", 100)
                axis = params.get("axis", "z")
                shape = cq.Workplane("XY").cylinder(
                    height, radius, centered=(True, True, False)
                )
                if axis == "x":
                    shape = shape.rotate((0, 0, 0), (0, 1, 0), 90)
                elif axis == "y":
                    shape = shape.rotate((0, 0, 0), (1, 0, 0), -90)
                shape = shape.translate((cx, cy, cz))
            elif op_type == "sphere":
                cx = params.get("cx", 0)
                cy = params.get("cy", 0)
                cz = params.get("cz", 0)
                radius = params.get("radius", 50)
                shape = cq.Workplane("XY").sphere(radius).translate((cx, cy, cz))
            else:
                logger.warning(f"Unknown CSG primitive type '{op_type}', skipping")
                continue

            # Apply operation
            if result is None or operation == "add":
                if result is None:
                    result = shape
                else:
                    result = result.union(shape)
            elif operation == "subtract":
                result = result.cut(shape)
            elif operation == "intersect":
                result = result.intersect(shape)

        if result is None:
            raise ValueError("No valid CSG operations produced geometry")

        return result

    def export_step(self, filepath: str) -> str:
        """Export the design domain to a STEP file."""
        import cadquery as cq

        geometry = self.build_geometry()
        cq.exporters.export(geometry, filepath)
        logger.info(f"Exported design domain to {filepath}")
        return filepath

    def get_bounding_box(self) -> Dict[str, float]:
        """
        Get the axis-aligned bounding box of the design domain.

        Returns:
            Dict with x_min, x_max, y_min, y_max, z_min, z_max
        """
        if self.shape_type == "box" and self.bounds:
            return self.bounds

        # For other shapes, compute from geometry
        geometry = self.build_geometry()
        bb = geometry.val().BoundingBox()

        return {
            "x_min": bb.xmin,
            "x_max": bb.xmax,
            "y_min": bb.ymin,
            "y_max": bb.ymax,
            "z_min": bb.zmin,
            "z_max": bb.zmax,
        }
