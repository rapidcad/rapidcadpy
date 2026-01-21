from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 2.25).line_to(2.165, 2.25).line_to(
    2.165, 2.9
).line_to(3.765, 2.9).line_to(3.765, 2.25).line_to(5.93, 2.25).line_to(
    5.93, 1.6
).line_to(
    12.63, 1.6
).line_to(
    12.63, 0.0
).line_to(
    0.0, 0.0
)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

# Sketch 2
wp2 = app.work_plane("XY")

wp2.move_to(0.38, 2.125).line_to(0.565, 2.125).line_to(0.565, 2.25).line_to(
    0.38, 2.25
).line_to(0.38, 2.125)

# Revolve feature 2
shape2 = wp2.revolve(6.283185307179586, "X", "Cut")

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(2.165, 2.45).line_to(2.185, 2.375359)
wp3.move_to(2.026962, 2.22).line_to(1.915, 2.25).line_to(1.915, 2.45).line_to(
    2.165, 2.45
)
wp3.move_to(2.026962, 2.22).three_point_arc((2.160525, 2.242195), (2.185, 2.375359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, "X", "Cut")

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(5.55, 2.125).line_to(5.365, 2.125).line_to(5.365, 2.25).line_to(
    5.55, 2.25
).line_to(5.55, 2.125)

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, "X", "Cut")

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(3.765, 2.45).line_to(3.745, 2.375359).three_point_arc(
    (3.769475, 2.242195), (3.903038, 2.22)
).line_to(4.015, 2.25).line_to(4.015, 2.45).line_to(3.765, 2.45)

# Revolve feature 5
shape5 = wp5.revolve(6.283185307179586, "X", "Cut")

# Sketch 6
wp6 = app.work_plane("XY", offset=1.6)

wp6.move_to(6.48, 0.0).line_to(12.08, 0.0)
wp6.move_to(6.98, -0.5).line_to(11.58, -0.5).three_point_arc(
    (12.08, 0.0), (11.58, 0.5)
).line_to(6.98, 0.5).three_point_arc((6.48, 0.0), (6.98, -0.5))

# Extrude feature 6
shape6 = wp6.extrude(-0.5, "Cut", symmetric=False)

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(5.93, 1.8).line_to(5.91, 1.725359).three_point_arc(
    (5.934475, 1.592195), (6.068038, 1.57)
).line_to(6.18, 1.6).line_to(6.18, 1.8).line_to(5.93, 1.8)

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, "X", "Cut")
