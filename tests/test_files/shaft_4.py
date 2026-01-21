from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 2.7).line_to(11.0, 2.7).line_to(11.0, 3.5).line_to(
    15.715, 3.5
).line_to(15.715, 4.2).line_to(17.915, 4.2).line_to(17.915, 3.5).line_to(
    20.63, 3.5
).line_to(
    20.63, 2.75
).line_to(
    22.23, 2.75
).line_to(
    22.23, 2.1
).line_to(
    32.73, 2.1
).line_to(
    32.73, 0.0
).line_to(
    0.0, 0.0
)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

# Sketch 2
wp2 = app.work_plane("XY", offset=2.7)

wp2.move_to(1.0, 0.0).line_to(10.0, 0.0)
wp2.move_to(1.8, -0.8).line_to(9.2, -0.8).three_point_arc(
    (10.0, 0.0), (9.2, 0.8)
).line_to(1.8, 0.8).three_point_arc((1.0, 0.0), (1.8, -0.8))

# Extrude feature 2
shape2 = wp2.extrude(-0.6, "Cut", symmetric=False)

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(11.0, 2.9).line_to(11.02, 2.825359)
wp3.move_to(10.861962, 2.67).line_to(10.75, 2.7).line_to(10.75, 2.9).line_to(11.0, 2.9)
wp3.move_to(10.861962, 2.67).three_point_arc((10.995525, 2.692195), (11.02, 2.825359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, "X", "Cut")

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(15.715, 3.7).line_to(15.735, 3.625359)
wp4.move_to(15.576962, 3.47).line_to(15.465, 3.5).line_to(15.465, 3.7).line_to(
    15.715, 3.7
)
wp4.move_to(15.576962, 3.47).three_point_arc((15.710525, 3.492195), (15.735, 3.625359))

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, "X", "Cut")

# Sketch 5
wp5 = app.work_plane("XY", offset=3.5)

wp5.move_to(10.6, 0.0).line_to(12.4, 0.0)
wp5.move_to(11.0, -0.4).line_to(12.0, -0.4).three_point_arc(
    (12.4, 0.0), (12.0, 0.4)
).line_to(11.0, 0.4).three_point_arc((10.6, 0.0), (11.0, -0.4))

# Extrude feature 5
shape5 = wp5.extrude(-0.35, "Cut", symmetric=False)

# Sketch 6
wp6 = app.work_plane("XY")

wp6.move_to(20.18, 3.35).line_to(19.915, 3.35).line_to(19.915, 3.5).line_to(
    20.18, 3.5
).line_to(20.18, 3.35)

# Revolve feature 6
shape6 = wp6.revolve(6.283185307179586, "X", "Cut")

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(17.915, 3.7).line_to(17.895, 3.625359).three_point_arc(
    (17.919475, 3.492195), (18.053038, 3.47)
).line_to(18.165, 3.5).line_to(18.165, 3.7).line_to(17.915, 3.7)

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, "X", "Cut")

# Sketch 8
wp8 = app.work_plane("XY")

wp8.move_to(20.63, 2.95).line_to(20.61, 2.875359).three_point_arc(
    (20.634475, 2.742195), (20.768038, 2.72)
).line_to(20.88, 2.75).line_to(20.88, 2.95).line_to(20.63, 2.95)

# Revolve feature 8
shape8 = wp8.revolve(6.283185307179586, "X", "Cut")

# Sketch 9
wp9 = app.work_plane("XY", offset=2.1)

wp9.move_to(22.98, 0.0).line_to(31.98, 0.0)
wp9.move_to(23.58, -0.6).line_to(31.38, -0.6).three_point_arc(
    (31.98, 0.0), (31.38, 0.6)
).line_to(23.58, 0.6).three_point_arc((22.98, 0.0), (23.58, -0.6))

# Extrude feature 9
shape9 = wp9.extrude(-0.5, "Cut", symmetric=False)

# Sketch 10
wp10 = app.work_plane("XY")

wp10.move_to(22.23, 2.3).line_to(22.21, 2.225359).three_point_arc(
    (22.234475, 2.092195), (22.368038, 2.07)
).line_to(22.48, 2.1).line_to(22.48, 2.3).line_to(22.23, 2.3)

# Revolve feature 10
shape10 = wp10.revolve(6.283185307179586, "X", "Cut")
