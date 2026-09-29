"""Create and export a simple plate with a through-hole."""

from pathlib import Path

from rapidcadpy import OpenCascadeOcpApp


output = Path("quickstart.step")
app = OpenCascadeOcpApp()

plate = app.work_plane("XY").rect(60, 40, centered=True).close().extrude(8)
hole = app.work_plane("XY").circle(6).close().extrude(8)
result = plate.cut(hole)
result.to_step(str(output))

print(f"Wrote {output} ({result.volume():.1f} mm³)")
