import os
from typing import TYPE_CHECKING, Optional

from rapidcadpy.app import App
from rapidcadpy.cad_types import VectorLike

if TYPE_CHECKING:
    from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane


class InventorApp(App):
    def __init__(self, headless: bool = True):
        try:
            import win32com.client as win32
            from win32com.client import Dispatch, gencache
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        super().__init__(InventorWorkPlane)
        try:
            self.inventor_app = win32.GetActiveObject("Inventor.Application")
        except Exception:
            self.inventor_app = Dispatch("Inventor.Application")
            self.inventor_app.Visible = not headless

        # Load the Inventor COM wrapper (type lib) if available
        self.mod = None
        try:
            ensure_module = getattr(gencache, "EnsureModule", None)
            if ensure_module is not None:
                self.mod = ensure_module(
                    "{D98A091D-3A0F-4C3E-B36E-61F62068D488}", 0, 1, 0
                )
        except Exception:
            self.mod = None

        self.transient_geom = self.inventor_app.TransientGeometry
        self.transient_obj = self.inventor_app.TransientObjects

    def new_document(self):
        import win32com.client as win32

        # Create a new PartDocument
        self.inventor_document = self.inventor_app.Documents.Add(
            win32.constants.kPartDocumentObject, "", True
        )
        self.part_doc = win32.CastTo(self.inventor_document, "PartDocument")
        self.comp_def = self.part_doc.ComponentDefinition

    def open_document(self, file_path):
        import win32com.client as win32

        doc = self.inventor_app.Documents.Open(file_path)

        try:
            # Cast to proper PartDocument type using the COM wrapper if available
            if self.mod is not None:
                self.part_doc = self.mod.PartDocument(doc)
            else:
                self.part_doc = win32.CastTo(doc, "PartDocument")
            self.comp_def = self.part_doc.ComponentDefinition
            self.inventor_document = doc
            return self.part_doc
        except Exception:
            raise TypeError(
                f"File '{file_path}' is not a valid Inventor Part Document."
            )

    def work_plane(
        self,
        name: str = "XY",
        origin: Optional[VectorLike] = None,
        normal: Optional[VectorLike] = None,
        offset: Optional[float] = None,
        **kwargs,
    ) -> "InventorWorkPlane":
        """
        Create a workplane either by name or by origin and normal vector.

        Args:
            name: Standard plane name ("XY", "XZ", "YZ") - used if origin/normal not provided
            origin: Origin point for custom workplane
            normal: Normal vector for custom workplane
            offset: Offset distance for the workplane
            **kwargs: Additional arguments (like app parameter)

        Returns:
            InventorWorkPlane instance
        """
        from rapidcadpy.integrations.inventor.workplane import InventorWorkPlane

        if origin is not None and normal is not None:
            # Create custom workplane from origin and normal
            return InventorWorkPlane.from_origin_normal(
                app=self, origin=origin, normal=normal
            )
        elif offset is not None and name in ["XY", "XZ", "YZ"]:
            # Create standard named workplane with offset
            return InventorWorkPlane.create_offset_plane(
                app=self, name=name, offset=offset
            )
        elif name == "XY":
            # Create standard XY workplane at origin
            return InventorWorkPlane.xy_plane(app=self)
        elif name == "XZ":
            return InventorWorkPlane.xz_plane(app=self)
        elif name == "YZ":
            return InventorWorkPlane.yz_plane(app=self)
        else:
            # Default to XY if unknown name
            return InventorWorkPlane.xy_plane(app=self)

    def to_stl(self, file_name: str) -> None:
        """
        Export the shape to STL format using Autodesk Inventor COM API.

        Uses the STL TranslatorAddIn with proper type casting to enable
        SaveCopyAs method with export options (resolution, format, units).

        Args:
            file_name: Path to the output STL file
        """
        try:
            import pythoncom
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Initialize COM library for this thread
        pythoncom.CoInitialize()

        try:
            # Ensure the file has .stl extension
            if not file_name.lower().endswith(".stl"):
                file_name += ".stl"

            # Convert to absolute path
            file_path = os.path.abspath(file_name)
            active_doc = self.inventor_app.ActiveDocument

            # Verify it's a part document
            if active_doc.DocumentType != win32.constants.kPartDocumentObject:
                raise RuntimeError(
                    "Active document must be a Part document for STL export"
                )

            # Get the STL Translator add-in (GUID is fixed for STL translator)
            stl_addin = self.inventor_app.ApplicationAddIns.ItemById(
                "{533E9A98-FC3B-11D4-8E7E-0010B541CD80}"
            )

            if stl_addin is None:
                raise RuntimeError("STL translator add-in not found in Inventor")

            # Cast to TranslatorAddIn using the type library module if available
            if self.mod is not None:
                stl_translator = self.mod.TranslatorAddIn(stl_addin)
            else:
                stl_translator = win32.CastTo(stl_addin, "TranslatorAddIn")

            # Create translation context and options
            ctx = self.transient_obj.CreateTranslationContext()
            ctx.Type = win32.constants.kFileBrowseIOMechanism

            # Create options map to check if translator supports this document
            opts_check = self.transient_obj.CreateNameValueMap()

            # Check if translator has save options for this document
            if stl_translator.HasSaveCopyAsOptions(active_doc, ctx, opts_check):
                # Create a new NameValueMap for actual export options
                opts = self.transient_obj.CreateNameValueMap()
                opts.Add("Resolution", 2)  # 0=Low, 1=Medium, 2=High, 3=Custom
                opts.Add("BinaryFormat", True)  # Binary STL (more efficient)
                opts.Add("ExportUnits", 2)  # kMillimeterLengthUnits = 2

                # Create data medium for output file
                data = self.transient_obj.CreateDataMedium()
                data.FileName = file_path

                # Perform the export
                stl_translator.SaveCopyAs(active_doc, ctx, opts, data)
                print(f"Successfully exported STL to: {file_path}")
            else:
                raise RuntimeError(
                    "STL translator has no save options for this document"
                )

            # Verify the file was created
            if not os.path.exists(file_path):
                raise RuntimeError(f"STL file was not created at: {file_path}")

        except Exception as e:
            # Re-raise with more context if it's not already a RuntimeError
            if not isinstance(e, RuntimeError):
                raise RuntimeError(f"Failed to export STL: {str(e)}")
            raise
        finally:
            # Uninitialize COM library
            pythoncom.CoUninitialize()

    def to_ipt(self, file_name: str) -> None:
        """
        Save the current document as an Inventor Part (.ipt) file using Autodesk Inventor COM API.

        Args:
            file_name: Path to the output IPT file
        """
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        # Ensure the file has .ipt extension
        if not file_name.lower().endswith(".ipt"):
            file_name += ".ipt"

        # Convert to absolute path
        file_path = os.path.abspath(file_name)
        active_doc = self.inventor_app.ActiveDocument

        # Verify it's a part document
        if active_doc.DocumentType != win32.constants.kPartDocumentObject:
            raise RuntimeError("Active document must be a Part document for IPT save")

        try:
            # Save current document state to a new IPT file
            active_doc.SaveAs(file_path, False)  # False = don't save as copy
            print(f"Successfully saved IPT to: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save IPT: {str(e)}")

        # Verify the file was created
        if not os.path.exists(file_path):
            raise RuntimeError(f"IPT file was not created at: {file_path}")

    def close_document(self) -> None:
        """
        Force-close the current active document, discarding unsaved changes and suppressing UI prompts.
        """
        try:
            active_doc = self.inventor_app.ActiveDocument
        except Exception:
            return
        if active_doc is None:
            return

        # Suppress UI prompts if supported
        old_silent = None
        had_silent = hasattr(self.inventor_app, "SilentOperation")
        if had_silent:
            try:
                old_silent = self.inventor_app.SilentOperation
                self.inventor_app.SilentOperation = True
            except Exception:
                had_silent = False

        try:
            # Close without saving changes
            active_doc.Close(False)
        finally:
            # Restore SilentOperation
            if had_silent:
                try:
                    self.inventor_app.SilentOperation = old_silent  # type: ignore[assignment]
                except Exception:
                    pass
            # Clear references
            try:
                self.part_doc = None  # type: ignore[assignment]
                self.comp_def = None  # type: ignore[assignment]
                self.inventor_document = None  # type: ignore[assignment]
            except Exception:
                pass

    def chamfer_edge(
        self,
        x: float,
        radius: float,
        angle: float,
        distance: float,
        tol: float = 1e-3,
    ) -> int:
        """Apply a distance-angle chamfer to edges that match a given X coordinate and radius.

        Searches all edges of all bodies and applies a Distance–Angle chamfer to
        those whose curve center.X and radius match within `tol`.

        Returns the number of chamfers created.
        """
        try:
            import win32com.client as win32

            # math not required if angle is already radians
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        if self.comp_def is None:
            raise RuntimeError(
                "No active Inventor part is available for chamfer operation."
            )

        constants = win32.constants

        # Helper to check if an edge's geometry is circular and matches x & radius
        def _edge_matches(e) -> bool:
            try:
                geom = e.Geometry
                center = getattr(geom, "Center", None)
                r = getattr(geom, "Radius", None)
                if center is None or r is None:
                    return False
                cx = getattr(center, "X", None)
                if cx is None:
                    return False
                return (abs(cx - x) <= tol) and (abs(r - radius) <= tol)
            except Exception:
                return False

        # Candidate faces: prioritize planar faces at given X (end caps), then other planars,
        # then cylindrical/conical as fallback (for revolved geometry)
        def _candidate_faces(e):
            try:
                faces = e.Faces
                exact_x_planars, other_planars, cylinders, cones, others = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i in range(1, faces.Count + 1):
                    f = faces.Item(i)
                    st = getattr(f, "SurfaceType", None)
                    if st == constants.kPlaneSurface:
                        try:
                            rb = f.Evaluator.RangeBox
                            minx = getattr(rb.MinPoint, "X", None)
                            maxx = getattr(rb.MaxPoint, "X", None)
                            if minx is not None and maxx is not None:
                                if (
                                    abs(minx - maxx) <= tol * 10
                                    and abs(minx - x) <= tol
                                ):
                                    exact_x_planars.append(f)
                                else:
                                    other_planars.append(f)
                            else:
                                other_planars.append(f)
                        except Exception:
                            other_planars.append(f)
                    elif st == constants.kCylinderSurface:
                        cylinders.append(f)
                    elif st == constants.kConeSurface:
                        cones.append(f)
                    else:
                        others.append(f)
                return exact_x_planars + other_planars + cylinders + cones + others
            except Exception:
                return []

        chamfer_features = self.comp_def.Features.ChamferFeatures
        created = 0

        # Prefer an EdgeCollection if available (per API signature), else ObjectCollection
        create_edge_collection = getattr(
            self.transient_obj, "CreateEdgeCollection", None
        )

        # Iterate edges across all bodies
        try:
            bodies = self.comp_def.SurfaceBodies
        except Exception as e:
            raise RuntimeError(f"Unable to access surface bodies: {e}")

        for b_idx in range(1, bodies.Count + 1):
            body = bodies.Item(b_idx)
            edges = body.Edges
            for e_idx in range(1, edges.Count + 1):
                edge = edges.Item(e_idx)
                if not _edge_matches(edge):
                    continue

                # Create the right kind of collection and add the edge
                coll = self.transient_obj.CreateEdgeCollection()
                coll.Add(edge)

                success = False
                for base_face in _candidate_faces(edge):
                    # Use radians for angle; try chain True then False
                    for chain in (True, False):
                        try:
                            chamfer_features.AddUsingDistanceAndAngle(
                                coll, base_face, float(distance), float(angle)
                            )
                            created += 1
                            success = True
                            break
                        except Exception:
                            continue
                    if success:
                        break
                # if not success, skip to next edge

        return created

    def add_thread(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        radius: Optional[float] = None,
        axis: Optional[str] = None,
        designation: str = "M8x1.25",
        thread_class: str = "6H",
        thread_type: str = "external",
        right_handed: bool = True,
        length: Optional[float] = None,
        full_length: bool = True,
        offset: float = 0.0,
        modeled: bool = False,
        tol: float = 1e-3,
    ) -> int:
        """Apply a thread to cylindrical faces that match given geometric properties.

        Searches all cylindrical faces and applies thread features to those matching
        the specified position, radius, and optionally axis direction.

        Args:
            x: X coordinate of the cylindrical face center (required if radius provided)
            y: Y coordinate of the cylindrical face center (optional, for better matching)
            z: Z coordinate of the cylindrical face center (optional, for better matching)
            radius: Radius of the cylindrical face to thread (required if x provided)
            axis: Cylinder axis direction ("X", "Y", or "Z") for better matching (optional)
            designation: Thread designation (e.g., "M8x1.25", "1/4-20 UNC")
            thread_class: Thread class/fit (e.g., "6H", "6g")
            thread_type: "internal" or "external"
            right_handed: True for right-hand thread, False for left-hand
            length: Thread length in cm (if None and full_length=False, uses face length)
            full_length: Whether thread spans the full face length
            offset: Offset from face start for thread placement
            modeled: True for modeled thread (actual geometry), False for cosmetic
            tol: Tolerance for matching position and radius

        Returns:
            Number of threads created

        Example:
            # External thread on a shaft at x=5, radius=0.4, along X-axis
            app.add_thread(x=5, radius=0.4, axis='X', designation="M8x1.25", thread_type="external")
            
            # Internal thread in a hole with full position specification
            app.add_thread(x=0, y=0, z=5, radius=0.35, axis='Z', 
                          designation="M7x1", thread_type="internal")
        """
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError(
                "pywin32 is required for Inventor integration. Install with: pip install pywin32"
            )

        if self.comp_def is None:
            raise RuntimeError(
                "No active Inventor part is available for thread operation."
            )

        constants = win32.constants
        thread_features = self.comp_def.Features.ThreadFeatures
        created = 0

        # Helper to check if a face is cylindrical and matches geometric properties
        def _face_matches(face) -> bool:
            try:
                # Check if it's a cylindrical surface
                if face.SurfaceType != constants.kCylinderSurface:
                    return False
                
                geom = face.Geometry
                base_pt = getattr(geom, "BasePoint", None)
                r = getattr(geom, "Radius", None)
                
                if base_pt is None or r is None:
                    return False
                
                # Match radius
                if not (abs(r - radius) <= tol):
                    return False
                
                # Match position (x is required, y/z optional for better precision)
                if not (abs(base_pt.X - x) <= tol):
                    return False
                
                if y is not None and not (abs(base_pt.Y - y) <= tol):
                    return False
                
                if z is not None and not (abs(base_pt.Z - z) <= tol):
                    return False
                
                # Match axis direction if specified
                if axis is not None:
                    axis_vec = getattr(geom, "AxisVector", None)
                    if axis_vec is not None:
                        if axis.upper() == "X" and abs(axis_vec.X) < 0.9:
                            return False
                        elif axis.upper() == "Y" and abs(axis_vec.Y) < 0.9:
                            return False
                        elif axis.upper() == "Z" and abs(axis_vec.Z) < 0.9:
                            return False
                
                return True
            except Exception:
                return False

        # Iterate through all faces in all bodies
        try:
            bodies = self.comp_def.SurfaceBodies
        except Exception as e:
            raise RuntimeError(f"Unable to access surface bodies: {e}")

        for b_idx in range(1, bodies.Count + 1):
            body = bodies.Item(b_idx)
            faces = body.Faces
            
            for f_idx in range(1, faces.Count + 1):
                face = faces.Item(f_idx)
                
                # If x and radius provided, match specific face
                if x is not None and radius is not None:
                    if not _face_matches(face):
                        continue
                
                # Skip if not cylindrical
                if face.SurfaceType != constants.kCylinderSurface:
                    continue

                family = "Unknown"
                try:
                    if modeled:
                        print(f"Warning: Modeled threads not yet supported, creating cosmetic thread instead")

                    is_internal = thread_type.lower() == "internal"
                    
                    # Fix thread class defaulting if it mismatches the type
                    # H is internal, g/h are external usually. 
                    # If user didn't specify class (it's the default "6H"), but type is external, switch to "6g".
                    if thread_class == "6H" and not is_internal:
                        thread_class = "6g"
                    
                    # Determine thread family (spreadsheet sheet name) based on designation
                    # Common defaults: "ISO Metric Profile" for metric, "ANSI Unified Screw Threads" for imperial
                    family = "ISO Metric Profile"
                    des_clean = designation.strip().upper()
                    if des_clean.startswith("M"):
                        family = "ISO Metric Profile"
                    elif ("-" in des_clean) or ("/" in des_clean) or ("UN" in des_clean):
                        family = "ANSI Unified Screw Threads"

                    # Create the thread info object first
                    thread_info = None
                    last_error = None
                    
                    # Strategies to try for thread creation
                    strategies = []
                    
                    # 1. Try the primary family derived from designation
                    strategies.append(family)
                    
                    # 2. Add fallbacks
                    if family == "ISO Metric Profile":
                        strategies.append("ANSI Metric M Profile")
                    elif family == "ANSI Unified Screw Threads":
                        strategies.append("Unified National")
                    
                    # 3. Always try the other common one as last resort
                    if "ISO Metric Profile" not in strategies:
                        strategies.append("ISO Metric Profile")
                        
                    for strategy_family in strategies:
                        try:
                            # Update family for logging
                            # family = strategy_family # KEEP ORIGINAL FAMILY FOR LOGGING? OR UPDATE?
                            
                            thread_info = thread_features.CreateStandardThreadInfo(
                                is_internal,  # Internal
                                right_handed, # RightHanded
                                strategy_family,       # ThreadType (Sheet name)
                                designation,  # ThreadDesignation
                                thread_class  # Class
                            )
                            # If successful, break
                            family = strategy_family
                            break
                        except Exception as e:
                            last_error = e
                            continue
                    
                    if thread_info is None:
                        if last_error:
                             print(f"Warning: Failed to create thread info for {designation} with families {strategies}: {last_error}")
                        
                        if last_error:
                            raise last_error
                        else:
                            raise RuntimeError(f"Could not create thread info for {designation}")

                    # Find a valid start edge from the face
                    if face.Edges.Count == 0:
                        print(f"Warning: Face {f_idx} has no edges, cannot add thread.")
                        continue
                     
                    # Heuristic for start edge:
                    # - If external, we usually want the edge at the start of the cylinder?
                    # - If internal, same?
                    # Simple heuristic: Use start_edge = face.Edges.Item(1)
                    # And try both directions if it fails? Or assume Item(1) works?
                    # Let's try to be smarter if we have offset/length logic, but for full length, Item(1) is fine.
                    
                    start_edge = face.Edges.Item(1)

                    # Add the thread feature using the info object
                    # Signature per docs:
                    # ThreadFeatures.Add(Face, StartEdge, ThreadInfo, [DirectionReversed], [FullDepth], [ThreadDepth], [ThreadOffset])
                    
                    try:
                        # Note: We pass positional arguments as much as possible to avoid keyword issues with COM wrapper
                        if full_length:
                             # Face, StartEdge, ThreadInfo, DirectionReversed, FullDepth
                             thread_feature = thread_features.Add(
                                face, 
                                start_edge, 
                                thread_info, 
                                False, 
                                True
                            )
                        else:
                            # Face, StartEdge, ThreadInfo, DirectionReversed, FullDepth, ThreadDepth, ThreadOffset
                            depth = length if length is not None else 1.0 
                            # If win32com wrapper is strict, we might need to be careful with optional args.
                            # But usually positional works.
                            thread_feature = thread_features.Add(
                                face, 
                                start_edge,
                                thread_info, 
                                False, # DirectionReversed
                                False, # FullDepth
                                float(depth), 
                                float(offset)
                            )
                        
                        created += 1
                        
                        if x is not None and radius is not None:
                            break
                    except Exception as e:
                        print(f"Warning: Failed to create thread feature on face {f_idx} (Attempt 1): {e}")
                        # Could try reversing direction or picking another edge?
                        continue
                    
                except Exception as e:
                    print(f"Warning: Failed to add thread to face {f_idx} (Designation: {designation}, Family: {family}): {e}")
                    continue

        return created