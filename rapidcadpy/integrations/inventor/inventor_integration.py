import math
import os
from typing import Any, Optional

import win32com.client as win32
from win32com.client import Dispatch, constants, gencache

from .cad import Cad
from .cad_types import Vector, Vertex
from .extrude import Extrude
from .integrations.base_integration import BaseIntegration
from .machining_feature import (
    CounterSunkHole,
    MachiningFeature,
    ParallelKeyway,
)
from .primitive import Arc, Circle, Line
from .sketch import Sketch
from .sketch_extrude import Extrude
from .wire import Wire


class InventorIntegration(BaseIntegration):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.inventor_app = win32.GetActiveObject("Inventor.Application")
        except Exception:
            self.inventor_app = Dispatch("Inventor.Application")
            self.inventor_app.Visible = True

        # Constants, FileManager, TransientGeometry as attributes
        self.k = win32.constants
        self.fm = self.inventor_app.FileManager
        self.tg = self.inventor_app.TransientGeometry
        self.to = self.inventor_app.TransientObjects

        # Load the Inventor COM wrapper (type lib)
        self.mod = gencache.EnsureModule(
            "{D98A091D-3A0F-4C3E-B36E-61F62068D488}", 0, 1, 0
        )

        # Don't create a document in __init__ - create it when needed
        self.part_doc = None
        self.comp_def = None
        self.inventor_document = None

        self.point2d_dict = {}
        self.sketchpoint_dict = {}

    def _ensure_document(self):
        """Create a new document if one doesn't exist."""
        if self.part_doc is None:
            # Create a new PartDocument
            self.inventor_document = self.inventor_app.Documents.Add(
                self.k.kPartDocumentObject, "", True
            )
            self.part_doc = self.mod.PartDocument(self.inventor_document)
            self.comp_def = self.part_doc.ComponentDefinition
            print("Created new Inventor document:", self.part_doc.DisplayName)

    def _get_point2d(self, x, y):
        key = (round(x, 6), round(y, 6))
        if key not in self.point2d_dict:
            self.point2d_dict[key] = self.tg.CreatePoint2d(x, y)
        return self.point2d_dict[key]

    def _get_sketch_point(self, x, y):
        key = (round(x, 6), round(y, 6))
        if key not in self.sketchpoint_dict:
            pt2d = self._get_point2d(x, y)
            self.sketchpoint_dict[key] = sketch.SketchPoints.Add(pt2d)
        return self.sketchpoint_dict[key]

    def get_name(self) -> str:
        """Get the name of this integration."""
        return "Inventor"

    def export_cad(self, cad, filename: Optional[str] = None, **kwargs) -> Any:
        """
        Export CAD to Inventor format with enhanced saving options.

        Args:
            cad: CAD object to export
            filename: Optional filename to save (.ipt file)
            **kwargs: Additional options:
                - close_on_save (bool): Close document after saving (default: False)
                - save_as_copy (bool): Save as copy without changing current document (default: False)
                - overwrite (bool): Overwrite existing file if it exists (default: True)
                - visible (bool): Make Inventor application visible (default: current state)

        Returns:
            Inventor part document object
        """
        close_on_save = kwargs.get("close_on_save", False)
        save_as_copy = kwargs.get("save_as_copy", False)
        overwrite = kwargs.get("overwrite", True)
        visible = kwargs.get("visible", None)

        # Set Inventor visibility if specified
        if visible is not None:
            self.inventor_app.Visible = visible

        # Export to Inventor
        result = self.to_inventor(cad, filename, close_on_save)

        # Enhanced saving logic
        if filename:
            # Ensure .ipt extension
            if not filename.lower().endswith(".ipt"):
                filename = filename + ".ipt"

            if os.path.exists(filename) and not overwrite:
                raise FileExistsError(
                    f"File {filename} already exists and overwrite=False"
                )

            try:
                if save_as_copy:
                    # Save as copy without changing the current document
                    self.inventor_document.SaveAs(filename, True)  # True = SaveCopyAs
                    print(f"✓ Saved copy to: {filename}")
                else:
                    # Regular save
                    self.inventor_document.SaveAs(filename, False)
                    print(f"✓ Saved Inventor file: {filename}")

                if close_on_save:
                    self.inventor_document.Close()
                    print("✓ Closed Inventor document")

            except Exception as e:
                print(f"Warning: Could not save file {filename}: {e}")

        return result

    def save_ipt(self, cad, filename: str, **kwargs) -> bool:
        """
        Convenience method specifically for saving .ipt files.

        Args:
            cad: CAD object to export
            filename: Filename for the .ipt file (extension added automatically)
            **kwargs: Same options as export_cad

        Returns:
            True if save was successful, False otherwise
        """
        try:
            self.export_cad(cad, filename, **kwargs)
            return True
        except Exception as e:
            print(f"Failed to save .ipt file: {e}")
            return False

    def to_inventor(
        self, cad, filename: Optional[str] = None, close_on_save: bool = False
    ):
        # Ensure we have a document to work with
        self._ensure_document()

        comp_def = self.part_doc.ComponentDefinition

        # Process each feature in the construction history
        for feature in cad.construction_history:
            if isinstance(feature, Extrude):
                # Handle sketch extrude features (existing logic)
                # Use the sketch_plane from the feature itself, not from individual sketches
                origin = feature.sketch_plane.origin
                x_dir = feature.sketch_plane.x_dir
                y_dir = feature.sketch_plane.y_dir
                z_dir = feature.sketch_plane.z_dir
                origin_pt = tg.CreatePoint(origin.x, origin.y, origin.z)
                x_vec = tg.CreateUnitVector(x_dir.x, x_dir.y, x_dir.z)
                y_vec = tg.CreateUnitVector(y_dir.x, y_dir.y, y_dir.z)
                # Add the work plane
                work_plane = comp_def.WorkPlanes.AddFixed(
                    origin_pt, x_vec, y_vec, False
                )
                work_plane.Visible = False  # optional: hide it

                # Now create the sketch on that custom plane
                sketch = comp_def.Sketches.Add(work_plane)

                # Process all sketches in the feature on the same plane
                for sketch_obj in feature.sketch:

                    def parse_edge_to_inventor(edge):
                        if isinstance(edge, Line):
                            sp = get_sketch_point(
                                edge.start_point.x, edge.start_point.y
                            )
                            ep = get_sketch_point(edge.end_point.x, edge.end_point.y)
                            sketch.SketchLines.AddByTwoPoints(sp, ep)
                        elif isinstance(edge, Circle):
                            center = tg.CreatePoint2d(edge.center.x, edge.center.y)
                            sketch.SketchCircles.AddByCenterRadius(center, edge.radius)
                        elif isinstance(edge, Arc):
                            p1 = get_sketch_point(
                                edge.start_point.x, edge.start_point.y
                            )
                            p2 = get_point2d(edge.mid_point.x, edge.mid_point.y)
                            p3 = get_sketch_point(edge.end_point.x, edge.end_point.y)
                            sketch.SketchArcs.AddByThreePoints(p1, p2, p3)
                        else:
                            print(f"Unsupported primitive: {type(edge)}")

                    for edge in sketch_obj.outer_wire.edges:
                        parse_edge_to_inventor(edge)
                    for inner_wire in sketch_obj.inner_wires:
                        for edge in inner_wire.edges:
                            parse_edge_to_inventor(edge)

                # Create the extrude after processing all sketches
                profile = sketch.Profiles.AddForSolid()
                ext_op = {
                    "JoinBodyFeatureOperation": 1,
                    "Cut": 2,
                    "Intersect": 3,
                    "NewBodyFeatureOperation": constants.kNewBodyOperation,
                }.get(feature.extrude.operation, 0)
                ext_def = comp_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(
                    profile, ext_op
                )
                ext_def.SetDistanceExtent(
                    feature.extrude.extent_one, constants.kPositiveExtentDirection
                )
                comp_def.Features.ExtrudeFeatures.Add(ext_def)

            elif isinstance(feature, MachiningFeature):
                # Handle machining features
                if isinstance(feature, CounterSunkHole):
                    # For now, create a simple sketch-based hole representation
                    origin = feature.sketch_plane.origin
                    x_dir = feature.sketch_plane.x_dir
                    y_dir = feature.sketch_plane.y_dir
                    z_dir = feature.sketch_plane.z_dir
                    origin_pt = tg.CreatePoint(origin.x, origin.y, origin.z)
                    x_vec = tg.CreateUnitVector(x_dir.x, x_dir.y, x_dir.z)
                    y_vec = tg.CreateUnitVector(y_dir.x, y_dir.y, y_dir.z)

                    # Add the work plane
                    work_plane = comp_def.WorkPlanes.AddFixed(
                        origin_pt, x_vec, y_vec, False
                    )
                    work_plane.Visible = False

                    # Create sketch for the hole
                    sketch = comp_def.Sketches.Add(work_plane)
                    center = tg.CreatePoint2d(
                        0.0, 0.0
                    )  # Center at origin of sketch plane
                    sketch.SketchCircles.AddByCenterRadius(
                        center, feature.diameter / 2.0
                    )

                    # Create extrude for the hole (cut operation)
                    profile = sketch.Profiles.AddForSolid()
                    ext_def = comp_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(
                        profile, constants.kCutOperation
                    )
                    ext_def.SetDistanceExtent(
                        feature.depth, constants.kPositiveExtentDirection
                    )
                    comp_def.Features.ExtrudeFeatures.Add(ext_def)

                elif isinstance(feature, ParallelKeyway):
                    # For now, create a simple sketch-based keyway representation
                    origin = feature.sketch_plane.origin
                    x_dir = feature.sketch_plane.x_dir
                    y_dir = feature.sketch_plane.y_dir
                    z_dir = feature.sketch_plane.z_dir
                    origin_pt = tg.CreatePoint(origin.x, origin.y, origin.z)
                    x_vec = tg.CreateUnitVector(x_dir.x, x_dir.y, x_dir.z)
                    y_vec = tg.CreateUnitVector(y_dir.x, y_dir.y, y_dir.z)

                    # Add the work plane
                    work_plane = comp_def.WorkPlanes.AddFixed(
                        origin_pt, x_vec, y_vec, False
                    )
                    work_plane.Visible = False

                    # Create sketch for the keyway
                    sketch = comp_def.Sketches.Add(work_plane)

                    # Create rectangle for keyway
                    if feature.orientation == "horizontal":
                        width, height = feature.length, feature.width
                    else:
                        width, height = feature.width, feature.length

                    # Create rectangle centered at origin
                    p1 = sketch.SketchPoints.Add(
                        tg.CreatePoint2d(-width / 2, -height / 2)
                    )
                    p2 = sketch.SketchPoints.Add(
                        tg.CreatePoint2d(width / 2, -height / 2)
                    )
                    p3 = sketch.SketchPoints.Add(
                        tg.CreatePoint2d(width / 2, height / 2)
                    )
                    p4 = sketch.SketchPoints.Add(
                        tg.CreatePoint2d(-width / 2, height / 2)
                    )

                    sketch.SketchLines.AddByTwoPoints(p1, p2)
                    sketch.SketchLines.AddByTwoPoints(p2, p3)
                    sketch.SketchLines.AddByTwoPoints(p3, p4)
                    sketch.SketchLines.AddByTwoPoints(p4, p1)

                    # Create extrude for the keyway (cut operation)
                    profile = sketch.Profiles.AddForSolid()
                    ext_def = comp_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(
                        profile, constants.kCutOperation
                    )
                    ext_def.SetDistanceExtent(
                        feature.depth, constants.kPositiveExtentDirection
                    )
                    comp_def.Features.ExtrudeFeatures.Add(ext_def)
                else:
                    print(f"Unsupported machining feature: {type(feature)}")
            else:
                print(f"Unsupported feature type: {type(feature)}")

        if filename:
            assert filename.endswith(".ipt")

            # Convert to absolute path
            import os

            abs_filename = os.path.abspath(filename)

            try:
                # Try to save the file
                self.inventor_document.SaveAs(abs_filename, False)
                print(f"✓ Saved Inventor file: {abs_filename}")

                if close_on_save:
                    self.inventor_document.Close()
                    print("✓ Closed Inventor document")

            except Exception as e:
                print(f"Warning: Could not save file {abs_filename}: {e}")
                print("The model was created in Inventor but not saved to file.")
                # Don't raise the exception - the model is still created in Inventor

    @staticmethod
    def from_inventor(filename: str) -> "Cad":
        """
        Read an Autodesk Inventor .ipt file and create a Cad object by extracting
        the construction history of sketches and extrude features.

        Args:
            filename: Path to the .ipt file

        Returns:
            Cad object reconstructed from the Inventor file
        """

        assert filename.endswith(".ipt"), "File must be an Inventor part file (.ipt)"

        # Connect to Inventor
        try:
            inventor_app = win32.GetActiveObject("Inventor.Application")
        except:
            inventor_app = Dispatch("Inventor.Application")
            inventor_app.Visible = True

        # Load the Inventor COM wrapper
        mod = gencache.EnsureModule("{D98A091D-3A0F-4C3E-B36E-61F62068D488}", 0, 1, 0)

        # Open the document
        inventor_document = inventor_app.Documents.Open(filename)
        part_doc = mod.PartDocument(inventor_document)
        comp_def = part_doc.ComponentDefinition

        construction_sequence = []

        try:
            # Iterate through all extrude features
            extrude_features = comp_def.Features.ExtrudeFeatures

            for i in range(1, extrude_features.Count + 1):
                extrude_feature = extrude_features.Item(i)

                # Extract extrude information
                extent = extrude_feature.Extent
                if hasattr(extent, "Distance"):
                    extent_value = float(extent.Distance.Value)
                else:
                    extent_value = 1.0  # Default value

                # Map operation type
                operation_map = {
                    constants.kJoinOperation: "JoinFeatureOperation",
                    constants.kCutOperation: "CutFeatureOperation",
                    constants.kIntersectOperation: "IntersectFeatureOperation",
                    constants.kNewBodyOperation: "NewBodyFeatureOperation",
                }

                operation = operation_map.get(
                    extrude_feature.Operation, "NewBodyFeatureOperation"
                )

                # Create extrude object
                extrude_obj = Extrude(
                    extent_one=extent_value,
                    extent_two=0.0,
                    direction=1,
                    operation=operation,
                )

                # Extract sketches from the profile
                profile = extrude_feature.Profile
                sketches = []

                # Process each sketch in the profile
                for j in range(1, profile.Count + 1):
                    profile_path = profile.Item(j)
                    sketch_obj = profile_path.SketchEntity.Parent

                    # Extract sketch plane information
                    plane_def = sketch_obj.PlanarEntity
                    origin = Vector(0.0, 0.0, 0.0)  # Default to XY plane
                    x_dir = Vector(1.0, 0.0, 0.0)
                    y_dir = Vector(0.0, 1.0, 0.0)
                    z_dir = Vector(0.0, 0.0, 1.0)

                    sketch_plane = Plane(
                        origin=origin, x_dir=x_dir, y_dir=y_dir, z_dir=z_dir
                    )

                    # Extract sketch geometry
                    wire_edges = []

                    # Process sketch lines
                    for k in range(1, sketch_obj.SketchLines.Count + 1):
                        line = sketch_obj.SketchLines.Item(k)
                        start_pt = line.StartSketchPoint.Geometry
                        end_pt = line.EndSketchPoint.Geometry

                        line_obj = Line(
                            start_point=Vertex(start_pt.X, start_pt.Y),
                            end_point=Vertex(end_pt.X, end_pt.Y),
                        )
                        wire_edges.append(line_obj)

                    # Process sketch circles
                    for k in range(1, sketch_obj.SketchCircles.Count + 1):
                        circle = sketch_obj.SketchCircles.Item(k)
                        center_pt = circle.CenterSketchPoint.Geometry
                        radius = circle.Radius

                        circle_obj = Circle(
                            center=Vertex(center_pt.X, center_pt.Y), radius=radius
                        )
                        wire_edges.append(circle_obj)

                    # Process sketch arcs
                    for k in range(1, sketch_obj.SketchArcs.Count + 1):
                        arc = sketch_obj.SketchArcs.Item(k)
                        start_pt = arc.StartSketchPoint.Geometry
                        end_pt = arc.EndSketchPoint.Geometry
                        center_pt = arc.CenterSketchPoint.Geometry

                        # Calculate mid point on arc
                        mid_angle = (arc.StartAngle + arc.EndAngle) / 2
                        mid_x = center_pt.X + arc.Radius * math.cos(mid_angle)
                        mid_y = center_pt.Y + arc.Radius * math.sin(mid_angle)

                        arc_obj = Arc(
                            start_point=Vertex(start_pt.X, start_pt.Y),
                            end_point=Vertex(end_pt.X, end_pt.Y),
                            mid_point=Vertex(mid_x, mid_y),
                        )
                        wire_edges.append(arc_obj)

                    # Create wire and sketch objects
                    if wire_edges:
                        outer_wire = Wire(edges=wire_edges)
                        sketch_obj = Sketch(
                            outer_wire=outer_wire,
                            inner_wires=[],
                            sketch_plane=sketch_plane,
                        )
                        sketches.append(sketch_obj)

                # Create sketch extrude object
                if sketches:
                    sketch_extrude = Extrude(sketch=sketches, extrude=extrude_obj)
                    construction_sequence.append(sketch_extrude)

            # Close the document
            inventor_document.Close()

            return Cad(construction_sequence=construction_sequence)

        except Exception as e:
            # Ensure document is closed on error
            try:
                inventor_document.Close()
            except:
                pass
            raise e

    def export_file(self, cad, filename: str, **kwargs) -> bool:
        """Export a CAD object to Inventor format."""
        try:
            self.to_inventor(cad, filename, **kwargs)
            return True
        except Exception:
            return False

    def import_file(self, file_path: str) -> Optional["Cad"]:
        """Import a file from Inventor and create a CAD object."""
        try:
            return self.from_inventor(file_path)
        except Exception:
            return None

    def to_step(self, cad, filename: str) -> bool:
        """Export a CAD object to STEP format via Inventor."""
        # TODO: Implement STEP export via Inventor
        return False

    def to_stl(self, cad, filename: str) -> bool:
        """Export a CAD object to STL format via Inventor."""
        # TODO: Implement STL export via Inventor
        return False

    def is_available(self) -> bool:
        """Check if Inventor is available and can be used."""
        try:
            import win32com.client

            win32com.client.GetActiveObject("Inventor.Application")
            return True
        except:
            try:
                import win32com.client
                from win32com.client import Dispatch

                app = Dispatch("Inventor.Application")
                app.Quit()
                return True
            except:
                return False

    def to_orthographic_view(
        self,
        scale: float | None = 1.0,
        base_xy_cm: tuple[float, float] = (10.0, 12.0),
        dx_cm: float = 12.0,
        dy_cm: float = 10.0,
        template_path: str | None = None,
        export_pdf: bool = True,
        pdf_path: str | None = None,
    ) -> str | None:
        """
        Create front/top/right + ISO views on a new drawing from the active or given model doc.
        Returns the PDF path if exported, else None.
        """
        k = self.k
        inv = self.inventor_app
        tg = self.tg

        # 0) pick model document
        model_doc = (
            inv.ActiveDocument if inv.ActiveDocument is not None else self.part_doc
        )

        # 1) drawing template
        if template_path:
            tmpl = template_path
        else:
            tmpl = self.fm.GetTemplateFile(
                k.kDrawingDocumentObject,
                "",
                k.kDefaultSystemOfMeasure,
                k.kDefaultEnvironment,
                "",
            )

        # 2) create drawing + first sheet
        drw = inv.Documents.Add(k.kDrawingDocumentObject, tmpl, True)
        sheet = drw.Sheets.Item(1)

        # 3) scale (simple): fixed or auto (fit by RangeBox height)
        if scale is None:
            try:
                rb = model_doc.ComponentDefinition.RangeBox
                w = abs(rb.MaxPoint.X - rb.MinPoint.X)
                h = abs(rb.MaxPoint.Y - rb.MinPoint.Y)
                d = abs(rb.MaxPoint.Z - rb.MinPoint.Z)
                # crude auto-scale heuristic to fit A3-ish area: tweak as needed
                major = max(w, h, d)
                scale = 1.0 / 5.0 if major == 0 else min(1.0, 200.0 / major)
            except Exception:
                scale = 1.0 / 5.0

        # 4) base view (Front)
        px, py = base_xy_cm
        p_base = tg.CreatePoint2d(px, py)
        base = sheet.DrawingViews.AddBaseView(
            model_doc,
            p_base,
            scale,
            k.kFrontViewOrientation,
            k.kHiddenLineRemovedDrawingViewStyle,
            k.kPrecise,
        )

        # 5) projected views
        top_view = sheet.DrawingViews.AddProjectedView(
            base,
            tg.CreatePoint2d(p_base.X, p_base.Y + dy_cm),
            k.kHiddenLineRemovedDrawingViewStyle,
        )
        right_view = sheet.DrawingViews.AddProjectedView(
            base,
            tg.CreatePoint2d(p_base.X + dx_cm, p_base.Y),
            k.kHiddenLineRemovedDrawingViewStyle,
        )
        iso_view = sheet.DrawingViews.AddProjectedView(
            base,
            tg.CreatePoint2d(p_base.X + dx_cm * 0.8, p_base.Y + dy_cm * 0.8),
            k.kShadedDrawingViewStyle,
        )

        # 6) display tweaks
        base.ShowHiddenLines = False
        top_view.ShowHiddenLines = False
        right_view.ShowHiddenLines = False

        # 7) update
        drw.Update()

        # 8) optional PDF export
        if export_pdf:
            pdf_addin = inv.ApplicationAddIns.ItemById(
                "{0AC6FD96-2F4D-42CE-8BE0-8AEA580399E4}"
            )
            ctx = self.to.CreateNameValueMap()
            ctx.Add("All_Color_AS_Black", False)
            ctx.Add("Remove_Line_Weights", False)
            ctx.Add("Vector_Resolution", 400)
            ctx.Add("Sheet_Range", k.kAllSheets)
            out_pdf = (
                pdf_path
                if pdf_path
                else (model_doc.FullFileName.rsplit(".", 1)[0] + "_ORTHO.pdf")
            )
            pdf_addin.SaveCopyAs(drw, ctx, out_pdf)
            print("Saved:", out_pdf)
            return out_pdf

        return None

    def makeLine(self, start, end):
        """
        Create a line in Inventor using the COM API, or fallback to a Line primitive for compatibility.
        Args:
            start: Start point (Vertex)
            end: End point (Vertex)
        Returns:
            Line primitive (for now, can be extended to Inventor-specific object)
        """
        # For now, just return the Line primitive. In a full implementation, this would use Inventor's API.
        from .primitive import Line

        return Line(start_point=start, end_point=end)


def add_inventor_backend_to_cad():
    """Add the to_inventor method to the CAD class (legacy support)."""
    from ..cad import Cad

    def to_inventor(
        self,
        inventor_app=None,
        part_doc=None,
        filename: Optional[str] = None,
        close_on_save: bool = False,
    ):
        """
        Export this CAD object to Inventor.

        This method now uses the backend manager system for improved flexibility.

        Args:
            inventor_app: Inventor application (optional)
            part_doc: Inventor part document (optional)
            filename: Optional filename to save to
            close_on_save: Whether to close the document after saving
        """
        # Use the backend manager system
        from ..integrations import export_with_backend

        kwargs = {}
        if filename is not None:
            kwargs["filename"] = filename
        if close_on_save:
            kwargs["close_on_save"] = close_on_save
        return export_with_backend(self, "inventor", **kwargs)

    # Only add if not already present (backend_manager already adds it)
    if not hasattr(Cad, "to_inventor"):
        setattr(Cad, "to_inventor", to_inventor)


# Call this to add the method when the module is imported (legacy support)
add_inventor_backend_to_cad()
