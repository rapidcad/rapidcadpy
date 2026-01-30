from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties


def test_volume():
    # Create a simple box
    box = BRepPrimAPI_MakeBox(20, 10, 10).Shape()

    props = GProp_GProps()
    try:
        print("Attempting volume calculation (default)...")
        brepgprop_VolumeProperties(box, props)
        print(f"Volume: {props.Mass()}")
    except Exception as e:
        print(f"Default failed: {e}")

    try:
        print("Attempting volume calculation (UseTriangulation=True)...")
        # Signature: S, VProps, Eps, OnlyClosed, SkipShared
        # Wait, the traceback showed:
        # S: TopoDS_Shape
        # VProps: GProp_GProps
        # OnlyClosed: bool (optional, default to Standard_False)
        # SkipShared: bool (optional, default to Standard_False)
        # UseTriangulation: bool (optional, default to Standard_False)

        # Let's try passing arguments.
        brepgprop_VolumeProperties(box, props, False, False, True)
        print(f"Volume (Triangulation): {props.Mass()}")
    except Exception as e:
        print(f"Triangulation failed: {e}")


if __name__ == "__main__":
    test_volume()
