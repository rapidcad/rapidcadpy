from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 2.2).line_to(10.7, 2.2).line_to(10.7, 3.0).line_to(
    13.165, 3.0
).line_to(13.165, 3.65).line_to(15.365, 3.65).line_to(15.365, 3.0).line_to(
    17.83, 3.0
).line_to(17.83, 0.0).line_to(0.0, 0.0)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

# Sketch 2
wp2 = app.work_plane("XY", offset=2.2)

wp2.move_to(0.85, 0.0).line_to(9.85, 0.0)
wp2.move_to(1.45, -0.6).line_to(9.25, -0.6).three_point_arc(
    (9.85, 0.0), (9.25, 0.6)
).line_to(1.45, 0.6).three_point_arc((0.85, 0.0), (1.45, -0.6))

# Extrude feature 2
shape2 = wp2.extrude(-0.5, "Cut", symmetric=False)

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(10.7, 2.4).line_to(10.72, 2.325359)
wp3.move_to(10.561962, 2.17).line_to(10.45, 2.2).line_to(10.45, 2.4).line_to(10.7, 2.4)
wp3.move_to(10.561962, 2.17).three_point_arc((10.695525, 2.192195), (10.72, 2.325359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, "X", "Cut")

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(11.15, 2.85).line_to(11.365, 2.85).line_to(11.365, 3.0).line_to(
    11.15, 3.0
).line_to(11.15, 2.85)

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, "X", "Cut")

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(13.165, 3.2).line_to(13.185, 3.125359)
wp5.move_to(13.026962, 2.97).line_to(12.915, 3.0).line_to(12.915, 3.2).line_to(
    13.165, 3.2
)
wp5.move_to(13.026962, 2.97).three_point_arc((13.160525, 2.992195), (13.185, 3.125359))

# Revolve feature 5
shape5 = wp5.revolve(6.283185307179586, "X", "Cut")

# Sketch 6
wp6 = app.work_plane("XY")

wp6.move_to(17.38, 2.85).line_to(17.165, 2.85).line_to(17.165, 3.0).line_to(
    17.38, 3.0
).line_to(17.38, 2.85)

# Revolve feature 6
shape6 = wp6.revolve(6.283185307179586, "X", "Cut")

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(15.365, 3.2).line_to(15.345, 3.125359).three_point_arc(
    (15.369475, 2.992195), (15.503038, 2.97)
).line_to(15.615, 3.0).line_to(15.615, 3.2).line_to(15.365, 3.2)

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, "X", "Cut")
