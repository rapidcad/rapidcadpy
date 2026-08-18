"""FreeCAD document hydration and serialization for :class:`CadSession`.

Builds the backend-neutral semantic mirror (objects, geometry signatures,
dependency/containment trees, serialized properties) from a live FreeCAD
document while retaining the native ``DocumentObject`` handles.

Provided as a mixin so the session's generic state (``self.objects``,
``self.runtime_objects``, ``self.parameters`` …) stays owned by
``CadSession``; these methods only read and write that state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...cad_objects import CadDocument, CadFeature, CadObject
from ...session_records import SemanticObject


class FreeCADHydrationMixin:
    """FreeCAD-specific document hydration and value serialization."""

    def _geometry_signature(self, shape: Any) -> Dict[str, Any]:
        obj = getattr(shape, "obj", None)
        if obj is None:
            return {}
        return self._shape_signature(obj)

    def _hydrate_freecad_document(
        self,
        doc: Any,
        file_path: Optional[Path],
        source_tool: str = "open_document",
    ) -> Dict[str, Any]:
        """Build the semantic mirror while retaining native DocumentObjects."""
        self.active_workplane = None
        self.active_workplane_id = None
        self.active_shape_id = None
        self.objects.clear()
        self.runtime_objects.clear()
        self.operations.clear()
        self.parameters.clear()
        self.geometry_signatures.clear()

        native_objects = list(getattr(doc, "Objects", []))
        cad_document = self.cad_document
        if cad_document is None or cad_document.native_handle is not doc:
            cad_document = getattr(self.app, "cad_document", None)
        if cad_document is None or cad_document.native_handle is not doc:
            from rapidcadpy.integrations.freecad.cad_adapter import FreeCADAdapter

            adapter = FreeCADAdapter()
            cad_document = CadDocument(
                backend=adapter.backend_name,
                native_handle=doc,
                adapter=adapter,
                name=str(getattr(doc, "Name", "")),
                label=str(getattr(doc, "Label", "")),
                file_name=str(getattr(doc, "FileName", "") or file_path or ""),
            )
        self.cad_document = cad_document
        parameter_adapter = getattr(cad_document.adapter, "parameter_adapter", None)

        ids_by_name: Dict[str, str] = {}
        for native_obj in native_objects:
            native_name = str(native_obj.Name)
            ids_by_name[native_name] = cad_document.object_id(
                native_name,
                lambda: self._new_id("native"),
            )

        containment = self._freecad_containment(native_objects, ids_by_name)
        rapidcad_ids = list(ids_by_name.values())
        source_path = str(file_path) if file_path is not None else ""
        op_id = self._record(
            source_tool,
            {"path": source_path} if source_path else {},
            rapidcad_ids,
            f"Hydrated {len(native_objects)} native FreeCAD objects",
        )

        for native_obj in native_objects:
            rapidcad_id = ids_by_name[str(native_obj.Name)]
            semantic_type = self._freecad_semantic_type(native_obj)
            geometry: Dict[str, Any] = {}
            shape = getattr(native_obj, "Shape", None)
            if shape is not None and not getattr(shape, "isNull", lambda: False)():
                geometry = self._shape_signature(shape)
                self.geometry_signatures[rapidcad_id] = geometry

            depends_on = self._freecad_object_ids(
                getattr(native_obj, "OutList", []), ids_by_name
            )
            dependents = self._freecad_object_ids(
                getattr(native_obj, "InList", []), ids_by_name
            )
            properties = self._freecad_properties(native_obj, ids_by_name)
            capabilities = {"get_properties", "set_properties"}
            if geometry:
                capabilities.add("geometry")
            if semantic_type in {"feature", "sketch"}:
                if str(getattr(native_obj, "TypeId", "")) not in {
                    "Part::Feature",
                    "PartDesign::Feature",
                }:
                    capabilities.add("feature_history")
                wrapper_type = CadFeature
            else:
                wrapper_type = CadObject
            if parameter_adapter is not None:
                for binding_name in parameter_adapter.supported_feature_properties(
                    native_obj
                ):
                    capabilities.add(f"parameter.bind:{binding_name}")
            cad_object = wrapper_type(
                id=rapidcad_id,
                document=cad_document,
                native_handle=native_obj,
                native_name=str(getattr(native_obj, "Name", "")),
                native_type=str(getattr(native_obj, "TypeId", "")),
                label=str(getattr(native_obj, "Label", native_obj.Name)),
                semantic_type=semantic_type,
                visibility=getattr(
                    getattr(native_obj, "ViewObject", None), "Visibility", None
                ),
                properties=properties,
                depends_on=depends_on,
                dependents=dependents,
                parent_ids=containment["parents"].get(rapidcad_id, []),
                child_ids=containment["children"].get(rapidcad_id, []),
                geometry=geometry,
                capabilities=frozenset(capabilities),
            )
            self.runtime_objects[rapidcad_id] = cad_object
            metadata: Dict[str, Any] = {
                "native_name": cad_object.native_name,
                "native_type": cad_object.native_type,
                "backend": cad_object.backend,
                "visibility": cad_object.visibility,
                "properties": cad_object.properties,
                "dependencies": {
                    "depends_on": cad_object.depends_on,
                    "dependents": cad_object.dependents,
                },
                "parent_ids": cad_object.parent_ids,
                "child_ids": cad_object.child_ids,
                "capabilities": sorted(cad_object.capabilities),
            }
            if geometry:
                metadata["geometry"] = geometry
            self.objects[rapidcad_id] = SemanticObject(
                id=rapidcad_id,
                type=semantic_type,
                label=cad_object.label,
                source_op=op_id,
                source="native_document",
                metadata=metadata,
            )

        object_snapshots = [obj.to_dict() for obj in self.objects.values()]
        if parameter_adapter is not None:
            discovered_parameters = parameter_adapter.discover_parameters(
                doc,
                cad_document,
                ids_by_name,
                lambda: self._new_id("parameter"),
            )
            self.parameters = {
                parameter.id: parameter for parameter in discovered_parameters
            }
        parameter_snapshots = [
            parameter.to_dict() for parameter in self.parameters.values()
        ]
        revision_payload = json.dumps(
            {
                "objects": object_snapshots,
                "parameters": parameter_snapshots,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.document_revision = (
            "sha256:" + hashlib.sha256(revision_payload).hexdigest()
        )
        cad_document.revision = self.document_revision
        cad_document.name = str(getattr(doc, "Name", ""))
        cad_document.label = str(getattr(doc, "Label", ""))
        cad_document.file_name = str(getattr(doc, "FileName", "") or source_path)
        self.document = {
            key: value
            for key, value in cad_document.to_dict().items()
            if key != "backend"
        }
        tree = self._freecad_tree(
            rapidcad_ids, containment["children"], containment["parents"]
        )
        return self._ok(
            summary=(
                f"Hydrated active FreeCAD document with {len(native_objects)} objects"
                if source_tool == "use_active_document"
                else (
                    (
                        "Opened and hydrated FreeCAD document with "
                        f"{len(native_objects)} objects"
                    )
                    if source_tool == "open_document"
                    else (
                        f"Hydrated FreeCAD document after {source_tool} with "
                        f"{len(native_objects)} objects"
                    )
                )
            ),
            operation_id=op_id,
            document=dict(self.document),
            document_revision=self.document_revision,
            object_count=len(native_objects),
            objects=object_snapshots,
            parameter_count=len(parameter_snapshots),
            parameters=parameter_snapshots,
            tree=tree,
        )

    def _freecad_properties(
        self, obj: Any, ids_by_name: Dict[str, str]
    ) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        for name in getattr(obj, "PropertiesList", []):
            try:
                value = obj.getPropertyByName(name)
                if name == "Shape":
                    signature = self._shape_signature(value)
                    properties[name] = {
                        "kind": "geometry",
                        "signature": signature,
                    }
                else:
                    properties[name] = self._serialize_freecad_value(value, ids_by_name)
            except Exception as exc:
                properties[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
        return properties

    def _serialize_freecad_value(
        self, value: Any, ids_by_name: Dict[str, str], depth: int = 0
    ) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if depth >= 4:
            return str(value)
        native_name = getattr(value, "Name", None)
        if native_name is not None and str(native_name) in ids_by_name:
            return {"object_id": ids_by_name[str(native_name)]}
        if isinstance(value, (list, tuple)):
            return [
                self._serialize_freecad_value(item, ids_by_name, depth + 1)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                str(key): self._serialize_freecad_value(item, ids_by_name, depth + 1)
                for key, item in value.items()
            }
        if all(hasattr(value, attr) for attr in ("x", "y", "z")):
            return {
                "x": float(value.x),
                "y": float(value.y),
                "z": float(value.z),
            }
        if hasattr(value, "Base") and hasattr(value, "Rotation"):
            placement: Dict[str, Any] = {
                "base": self._serialize_freecad_value(
                    value.Base, ids_by_name, depth + 1
                )
            }
            try:
                placement["rotation_quaternion"] = [
                    float(component) for component in value.Rotation.Q
                ]
            except Exception:
                placement["rotation"] = str(value.Rotation)
            return placement
        if hasattr(value, "Value"):
            quantity: Dict[str, Any] = {"value": float(value.Value)}
            unit = str(getattr(value, "Unit", ""))
            if unit:
                quantity["unit"] = unit
            return quantity
        return str(value)

    def _freecad_object_ids(
        self, native_objects: Any, ids_by_name: Dict[str, str]
    ) -> list[str]:
        result = []
        for native_obj in native_objects or []:
            rapidcad_id = ids_by_name.get(str(getattr(native_obj, "Name", "")))
            if rapidcad_id is not None:
                result.append(rapidcad_id)
        return result

    def _freecad_containment(
        self, native_objects: list[Any], ids_by_name: Dict[str, str]
    ) -> Dict[str, Dict[str, list[str]]]:
        children: Dict[str, list[str]] = {item: [] for item in ids_by_name.values()}
        parents: Dict[str, list[str]] = {item: [] for item in ids_by_name.values()}
        for native_obj in native_objects:
            parent_id = ids_by_name[str(native_obj.Name)]
            members = getattr(native_obj, "Group", []) or []
            for child_id in self._freecad_object_ids(members, ids_by_name):
                if child_id not in children[parent_id]:
                    children[parent_id].append(child_id)
                if parent_id not in parents[child_id]:
                    parents[child_id].append(parent_id)

            try:
                geo_parent = native_obj.getParentGeoFeatureGroup()
            except Exception:
                geo_parent = None
            if geo_parent is not None:
                geo_parent_id = ids_by_name.get(str(getattr(geo_parent, "Name", "")))
                if geo_parent_id and parent_id not in children[geo_parent_id]:
                    children[geo_parent_id].append(parent_id)
                    parents[parent_id].append(geo_parent_id)
        return {"children": children, "parents": parents}

    def _freecad_tree(
        self,
        object_ids: list[str],
        children: Dict[str, list[str]],
        parents: Dict[str, list[str]],
    ) -> list[Dict[str, Any]]:
        def node(object_id: str, ancestors: set[str]) -> Dict[str, Any]:
            if object_id in ancestors:
                return {"id": object_id, "cycle": True, "children": []}
            next_ancestors = set(ancestors)
            next_ancestors.add(object_id)
            return {
                "id": object_id,
                "children": [
                    node(item, next_ancestors) for item in children[object_id]
                ],
            }

        roots = [object_id for object_id in object_ids if not parents[object_id]]
        return [node(object_id, set()) for object_id in roots]

    def _freecad_semantic_type(self, obj: Any) -> str:
        type_id = str(getattr(obj, "TypeId", ""))
        if type_id == "App::Part":
            return "part"
        if type_id == "PartDesign::Body":
            return "body"
        if "Sketch" in type_id:
            return "sketch"
        if type_id.startswith(("Part::", "PartDesign::")):
            return "feature"
        return "native_object"

    def _describe_freecad_object(self, obj: Any) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "name": getattr(obj, "Name", None),
            "label": getattr(obj, "Label", None),
            "type_id": getattr(obj, "TypeId", None),
            "visibility": getattr(getattr(obj, "ViewObject", None), "Visibility", None),
        }
        shape = getattr(obj, "Shape", None)
        if shape is not None and not getattr(shape, "isNull", lambda: False)():
            item["geometry"] = self._shape_signature(shape)
        return item

    def _shape_signature(self, obj: Any) -> Dict[str, Any]:
        signature: Dict[str, Any] = {}
        try:
            bb = obj.BoundBox
            signature["bbox"] = {
                "x_min": float(bb.XMin),
                "x_max": float(bb.XMax),
                "y_min": float(bb.YMin),
                "y_max": float(bb.YMax),
                "z_min": float(bb.ZMin),
                "z_max": float(bb.ZMax),
                "x_length": float(bb.XLength),
                "y_length": float(bb.YLength),
                "z_length": float(bb.ZLength),
            }
        except Exception:
            pass
        for attr, key in (("Volume", "volume"), ("Area", "area")):
            try:
                signature[key] = float(getattr(obj, attr))
            except Exception:
                pass
        for attr, key in (
            ("Faces", "face_count"),
            ("Edges", "edge_count"),
            ("Vertexes", "vertex_count"),
        ):
            try:
                signature[key] = len(getattr(obj, attr))
            except Exception:
                pass
        return signature
