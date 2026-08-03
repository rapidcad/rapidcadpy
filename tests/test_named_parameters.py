"""Backend-neutral and FreeCAD named-parameter tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapidcadpy.cad_objects import CadDocument
from rapidcadpy.integrations.freecad.cad_adapter import FreeCADAdapter
from rapidcadpy.integrations.freecad.parameter_adapter import (
    FreeCADParameterAdapter,
)


class FakeQuantity:
    def __init__(self, value: float, unit: str):
        self.Value = float(value)
        self.Unit = unit


class FakeObject:
    def __init__(self, name: str, type_id: str):
        object.__setattr__(self, "Name", name)
        object.__setattr__(self, "Label", name)
        object.__setattr__(self, "TypeId", type_id)
        object.__setattr__(self, "PropertiesList", [])
        object.__setattr__(self, "ExpressionEngine", [])
        object.__setattr__(self, "_property_types", {})
        object.__setattr__(self, "_property_groups", {})
        object.__setattr__(self, "_expressions", {})

    def __setattr__(self, name, value):
        property_type = self._property_types.get(name)
        if property_type in {"App::PropertyLength", "App::PropertyAngle"}:
            raw_value, unit = str(value).split(maxsplit=1)
            value = FakeQuantity(float(raw_value), unit)
        object.__setattr__(self, name, value)

    def addProperty(self, property_type, name, group, description):
        del description
        self.PropertiesList.append(name)
        self._property_types[name] = property_type
        self._property_groups[name] = group
        defaults = {
            "App::PropertyFloat": 0.0,
            "App::PropertyInteger": 0,
            "App::PropertyBool": False,
            "App::PropertyString": "",
            "App::PropertyLength": FakeQuantity(0, "mm"),
            "App::PropertyAngle": FakeQuantity(0, "deg"),
        }
        object.__setattr__(self, name, defaults[property_type])

    def getPropertyByName(self, name):
        return getattr(self, name)

    def getGroupOfProperty(self, name):
        return self._property_groups.get(name, "")

    def getTypeIdOfProperty(self, name):
        return self._property_types.get(name, "")

    def setExpression(self, name, expression):
        if expression is None:
            self._expressions.pop(name, None)
        else:
            self._expressions[name] = expression
        self.ExpressionEngine = list(self._expressions.items())


class FakeDocument:
    def __init__(self):
        self.Name = "Parameters"
        self.Label = "Parameters"
        self.FileName = ""
        self.Objects = []
        self.transaction_log = []
        self.recompute_count = 0

    def getObject(self, name):
        return next((obj for obj in self.Objects if obj.Name == name), None)

    def addObject(self, type_id, name):
        obj = FakeObject(name, type_id)
        self.Objects.append(obj)
        return obj

    def openTransaction(self, label):
        self.transaction_log.append(("open", label))

    def commitTransaction(self):
        self.transaction_log.append(("commit", None))

    def abortTransaction(self):
        self.transaction_log.append(("abort", None))

    def recompute(self):
        self.recompute_count += 1


def test_freecad_parameter_adapter_creates_discovers_and_binds_native_parameter():
    document = FakeDocument()
    adapter = FreeCADParameterAdapter()
    target = document.addObject("Part::Extrusion", "Extrude")
    target.PropertiesList.append("LengthFwd")
    target.LengthFwd = FakeQuantity(10, "mm")

    with adapter.transaction(document, "Create height"):
        handle = adapter.create_parameter(
            document,
            name="height",
            parameter_type="length",
            value=30,
            unit="mm",
        )
        native_property = adapter.bind_parameter(
            handle,
            target,
            property_name="length",
            expression="height / 2",
            available_names={"height"},
        )

    cad_document = CadDocument(
        backend="freecad",
        native_handle=document,
        adapter=FreeCADAdapter(),
        name=document.Name,
    )
    parameters = adapter.discover_parameters(
        document,
        cad_document,
        {"Extrude": "native_1", "RapidCADParameters": "native_2"},
        iter(["parameter_1"]).__next__,
    )

    assert native_property == "LengthFwd"
    assert "length" in adapter.supported_feature_properties(target)
    assert target.ExpressionEngine == [("LengthFwd", "RapidCADParameters.height / 2")]
    assert len(parameters) == 1
    assert parameters[0].to_dict() == {
        "id": "parameter_1",
        "name": "height",
        "label": "height",
        "parameter_type": "length",
        "value": 30.0,
        "unit": "mm",
        "expression": None,
        "evaluated_value": 30.0,
        "source": "native_user_parameter",
        "backend": "freecad",
        "writable": True,
        "renamable": False,
        "deletable": False,
        "bindable": True,
        "dependencies": [],
        "dependents": [
            {
                "object_id": "native_1",
                "property_name": "length",
                "expression": "height / 2",
            }
        ],
    }
    assert document.transaction_log == [
        ("open", "Create height"),
        ("commit", None),
    ]
    assert document.recompute_count == 1

    rediscovered = adapter.discover_parameters(
        document,
        cad_document,
        {"Extrude": "native_1", "RapidCADParameters": "native_2"},
        lambda: pytest.fail("Stable parameter IDs must not call the ID factory again."),
    )

    assert rediscovered[0].id == parameters[0].id


def test_parameter_expression_rejects_unknown_names_and_unsafe_syntax():
    adapter = FreeCADParameterAdapter()

    expression, names = adapter.translate_expression(
        "width * 2 + clearance",
        {"width", "clearance"},
    )

    assert expression == ("RapidCADParameters.width * 2 + RapidCADParameters.clearance")
    assert names == {"width", "clearance"}
    with pytest.raises(ValueError, match="Unknown parameter"):
        adapter.translate_expression("missing + 1", {"width"})
    with pytest.raises(ValueError, match="arithmetic operators only"):
        adapter.translate_expression("__import__('os')", set())


def test_unsupported_generic_feature_binding_reports_available_properties():
    adapter = FreeCADParameterAdapter()
    target = SimpleNamespace(
        TypeId="Part::Feature",
        PropertiesList=["Shape"],
    )

    with pytest.raises(NotImplementedError, match="Supported bindings: none"):
        adapter.resolve_feature_property(target, "length")
