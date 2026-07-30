"""Persistent named-parameter support for native FreeCAD documents."""

from __future__ import annotations

import ast
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from ...cad_objects import CadDocument, CadParameter, ParameterBinding


@dataclass(frozen=True)
class FreeCADParameterHandle:
    """Internal reference to a property on the native parameter container."""

    owner: Any
    property_name: str


class FreeCADParameterAdapter:
    """Store named parameters as typed properties in a FreeCAD document."""

    container_name = "RapidCADParameters"
    container_label = "RapidCAD Parameters"
    property_group = "RapidCAD Parameters"

    _PROPERTY_TYPES = {
        "length": "App::PropertyLength",
        "angle": "App::PropertyAngle",
        "number": "App::PropertyFloat",
        "integer": "App::PropertyInteger",
        "boolean": "App::PropertyBool",
        "string": "App::PropertyString",
    }
    _TYPE_BY_PROPERTY = {value: key for key, value in _PROPERTY_TYPES.items()}
    _DEFAULT_UNITS = {
        "length": "mm",
        "angle": "deg",
    }
    _FEATURE_PROPERTIES = {
        "Part::Extrusion": {
            "length": "LengthFwd",
            "reverse_length": "LengthRev",
            "taper_angle": "TaperAngle",
            "reverse_taper_angle": "TaperAngleRev",
        },
        "PartDesign::Pad": {
            "length": "Length",
            "second_length": "Length2",
            "taper_angle": "TaperAngle",
            "second_taper_angle": "TaperAngle2",
        },
        "Part::Revolution": {
            "angle": "Angle",
        },
        "PartDesign::Revolution": {
            "angle": "Angle",
        },
        "PartDesign::Hole": {
            "diameter": "Diameter",
            "depth": "Depth",
        },
        "PartDesign::Fillet": {
            "radius": "Radius",
        },
        "PartDesign::Chamfer": {
            "size": "Size",
        },
        "PartDesign::Thickness": {
            "thickness": "Value",
        },
    }

    @contextmanager
    def transaction(self, document: Any, label: str) -> Iterator[None]:
        """Apply a group of parameter edits as one native FreeCAD transaction."""
        opened = False
        try:
            if hasattr(document, "openTransaction"):
                document.openTransaction(label)
                opened = True
            yield
            document.recompute()
            if opened and hasattr(document, "commitTransaction"):
                document.commitTransaction()
        except Exception:
            if opened and hasattr(document, "abortTransaction"):
                document.abortTransaction()
            raise

    def ensure_container(self, document: Any) -> Any:
        container = self._document_object(document, self.container_name)
        if container is None:
            container = document.addObject("App::FeaturePython", self.container_name)
            container.Label = self.container_label
        return container

    def create_parameter(
        self,
        document: Any,
        *,
        name: str,
        parameter_type: str,
        value: Any,
        unit: Optional[str] = None,
        expression: Optional[str] = None,
        available_names: Iterable[str] = (),
    ) -> FreeCADParameterHandle:
        self.validate_name(name)
        normalized_type = parameter_type.strip().lower()
        property_type = self._PROPERTY_TYPES.get(normalized_type)
        if property_type is None:
            raise ValueError(
                f"Unsupported parameter_type '{parameter_type}'. Supported: "
                + ", ".join(sorted(self._PROPERTY_TYPES))
            )

        container = self.ensure_container(document)
        if name in getattr(container, "PropertiesList", []):
            raise ValueError(f"Named parameter '{name}' already exists.")
        container.addProperty(
            property_type,
            name,
            self.property_group,
            f"RapidCAD named {normalized_type} parameter",
        )
        handle = FreeCADParameterHandle(container, name)
        self.set_value(handle, value, unit=unit, parameter_type=normalized_type)
        if expression:
            known_names = set(available_names)
            known_names.add(name)
            self.set_expression(handle, expression, known_names)
        return handle

    def set_value(
        self,
        handle: FreeCADParameterHandle,
        value: Any,
        *,
        unit: Optional[str],
        parameter_type: str,
    ) -> None:
        owner = handle.owner
        if hasattr(owner, "setExpression"):
            owner.setExpression(handle.property_name, None)
        normalized_type = parameter_type.strip().lower()
        if normalized_type in {"length", "angle"}:
            resolved_unit = unit or self._DEFAULT_UNITS[normalized_type]
            setattr(owner, handle.property_name, f"{float(value)} {resolved_unit}")
        elif normalized_type == "number":
            setattr(owner, handle.property_name, float(value))
        elif normalized_type == "integer":
            setattr(owner, handle.property_name, int(value))
        elif normalized_type == "boolean":
            setattr(owner, handle.property_name, bool(value))
        elif normalized_type == "string":
            setattr(owner, handle.property_name, str(value))
        else:
            raise ValueError(f"Unsupported parameter type '{parameter_type}'.")

    def set_expression(
        self,
        handle: FreeCADParameterHandle,
        expression: str,
        available_names: Iterable[str],
    ) -> None:
        native_expression, dependencies = self.translate_expression(
            expression,
            available_names,
        )
        if handle.property_name in dependencies:
            raise ValueError(
                f"Parameter '{handle.property_name}' cannot depend on itself."
            )
        handle.owner.setExpression(handle.property_name, native_expression)

    def bind_parameter(
        self,
        handle: FreeCADParameterHandle,
        target: Any,
        property_name: str,
        expression: Optional[str],
        available_names: Iterable[str],
    ) -> str:
        native_property = self.resolve_feature_property(target, property_name)
        generic_expression = expression or handle.property_name
        native_expression, dependencies = self.translate_expression(
            generic_expression,
            available_names,
        )
        if handle.property_name not in dependencies:
            raise ValueError(
                f"Binding expression must reference parameter '{handle.property_name}'."
            )
        target.setExpression(native_property, native_expression)
        return native_property

    def discover_parameters(
        self,
        document: Any,
        cad_document: CadDocument,
        ids_by_name: Dict[str, str],
        id_factory: Callable[[], str],
    ) -> list[CadParameter]:
        container = self._document_object(document, self.container_name)
        if container is None:
            return []

        expression_entries = dict(self._expression_entries(container))
        result: list[CadParameter] = []
        parameter_names = [
            name
            for name in getattr(container, "PropertiesList", [])
            if self._property_group(container, name) == self.property_group
        ]
        for name in parameter_names:
            property_type = self._property_type(container, name)
            parameter_type = self._TYPE_BY_PROPERTY.get(property_type, "number")
            native_expression = expression_entries.get(name)
            expression = self._generic_expression(native_expression)
            dependencies = sorted(self.expression_dependencies(expression or ""))
            value, unit = self._value_and_unit(
                container.getPropertyByName(name),
                parameter_type,
            )
            dependents = self._find_dependents(
                document,
                ids_by_name,
                parameter_name=name,
            )
            result.append(
                CadParameter(
                    id=cad_document.parameter_id(name, id_factory),
                    document=cad_document,
                    native_handle=FreeCADParameterHandle(container, name),
                    name=name,
                    label=name,
                    parameter_type=parameter_type,
                    value=value,
                    unit=unit,
                    expression=expression,
                    evaluated_value=value,
                    source="native_user_parameter",
                    writable=True,
                    renamable=False,
                    deletable=not dependents,
                    bindable=True,
                    dependencies=dependencies,
                    dependents=dependents,
                )
            )
        return result

    def resolve_feature_property(self, target: Any, property_name: str) -> str:
        type_id = str(getattr(target, "TypeId", ""))
        mapping = self._FEATURE_PROPERTIES.get(type_id, {})
        native_property = mapping.get(property_name)
        if native_property is None:
            supported = ", ".join(sorted(mapping)) or "none"
            raise NotImplementedError(
                f"{type_id or 'Native object'} does not support generic parameter "
                f"binding '{property_name}'. Supported bindings: {supported}."
            )
        if native_property not in getattr(target, "PropertiesList", []):
            raise NotImplementedError(
                f"{type_id}.{native_property} is not available on this feature."
            )
        return native_property

    def supported_feature_properties(self, target: Any) -> list[str]:
        """Return generic parameter-binding names available on one feature."""
        mapping = self._FEATURE_PROPERTIES.get(
            str(getattr(target, "TypeId", "")),
            {},
        )
        native_properties = set(getattr(target, "PropertiesList", []))
        return sorted(
            generic_name
            for generic_name, native_name in mapping.items()
            if native_name in native_properties
        )

    def generic_feature_property(self, target: Any, native_property: str) -> str:
        mapping = self._FEATURE_PROPERTIES.get(
            str(getattr(target, "TypeId", "")),
            {},
        )
        for generic_name, candidate in mapping.items():
            if candidate == native_property:
                return generic_name
        return native_property

    @classmethod
    def validate_name(cls, name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(
                "Parameter names must start with a letter or underscore and "
                "contain only letters, numbers, and underscores."
            )

    @classmethod
    def translate_expression(
        cls,
        expression: str,
        available_names: Iterable[str],
    ) -> tuple[str, set[str]]:
        normalized = expression.strip()
        if not normalized:
            raise ValueError("Parameter expression must not be empty.")
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid parameter expression: {exc.msg}.") from exc

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.UAdd,
            ast.USub,
        )
        if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
            raise ValueError(
                "Parameter expressions support names, numeric literals, and "
                "arithmetic operators only."
            )

        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        available = set(available_names)
        unknown = names - available
        if unknown:
            raise ValueError(
                "Unknown parameter name(s) in expression: " + ", ".join(sorted(unknown))
            )

        class QualifyNames(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                return ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id=cls.container_name, ctx=ast.Load()),
                        attr=node.id,
                        ctx=node.ctx,
                    ),
                    node,
                )

        qualified = QualifyNames().visit(tree)
        ast.fix_missing_locations(qualified)
        return ast.unparse(qualified.body), names

    @staticmethod
    def expression_dependencies(expression: str) -> set[str]:
        if not expression:
            return set()
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return set()
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    @classmethod
    def _generic_expression(cls, expression: Optional[str]) -> Optional[str]:
        if not expression:
            return None
        return re.sub(
            rf"\b{re.escape(cls.container_name)}\.",
            "",
            str(expression),
        )

    def _find_dependents(
        self,
        document: Any,
        ids_by_name: Dict[str, str],
        *,
        parameter_name: str,
    ) -> list[ParameterBinding]:
        token = f"{self.container_name}.{parameter_name}"
        result: list[ParameterBinding] = []
        for obj in getattr(document, "Objects", []):
            if obj is self._document_object(document, self.container_name):
                continue
            object_id = ids_by_name.get(str(getattr(obj, "Name", "")))
            if object_id is None:
                continue
            for native_property, native_expression in self._expression_entries(obj):
                if token not in native_expression:
                    continue
                result.append(
                    ParameterBinding(
                        object_id=object_id,
                        property_name=self.generic_feature_property(
                            obj,
                            native_property,
                        ),
                        expression=self._generic_expression(native_expression)
                        or native_expression,
                    )
                )
        return result

    @staticmethod
    def _expression_entries(obj: Any) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for entry in getattr(obj, "ExpressionEngine", []) or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                result.append((str(entry[0]), str(entry[1])))
                continue
            path = getattr(entry, "Path", None)
            expression = getattr(entry, "Expression", None)
            if path is not None and expression is not None:
                result.append((str(path), str(expression)))
        return result

    @staticmethod
    def _document_object(document: Any, name: str) -> Any:
        getter = getattr(document, "getObject", None)
        if callable(getter):
            return getter(name)
        return next(
            (
                obj
                for obj in getattr(document, "Objects", [])
                if str(getattr(obj, "Name", "")) == name
            ),
            None,
        )

    @staticmethod
    def _property_group(obj: Any, name: str) -> str:
        try:
            return str(obj.getGroupOfProperty(name))
        except Exception:
            return ""

    @staticmethod
    def _property_type(obj: Any, name: str) -> str:
        try:
            return str(obj.getTypeIdOfProperty(name))
        except Exception:
            return ""

    def _value_and_unit(
        self,
        native_value: Any,
        parameter_type: str,
    ) -> tuple[Any, Optional[str]]:
        if parameter_type in {"length", "angle"} and hasattr(native_value, "Value"):
            unit = self._unit_name(native_value, parameter_type)
            return float(native_value.Value), unit
        if parameter_type == "number":
            return float(native_value), None
        if parameter_type == "integer":
            return int(native_value), None
        if parameter_type == "boolean":
            return bool(native_value), None
        if parameter_type == "string":
            return str(native_value), None
        return native_value, None

    def _unit_name(self, native_value: Any, parameter_type: str) -> str:
        raw = str(getattr(native_value, "Unit", "")).strip()
        match = re.search(r"Unit:\s*([^\s(]+)", raw)
        if match:
            return match.group(1)
        if raw and " " not in raw:
            return raw
        return self._DEFAULT_UNITS[parameter_type]
