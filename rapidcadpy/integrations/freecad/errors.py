"""FreeCAD integration errors."""


class FreeCADNativeFeatureError(RuntimeError):
    """Raised when a requested native FreeCAD feature cannot be created."""
