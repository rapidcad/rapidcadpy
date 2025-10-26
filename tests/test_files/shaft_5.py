from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 1.6).line_to(9.3, 1.6).line_to(9.3, 2.25).line_to(
    10.9, 2.25
).line_to(10.9, 2.75).line_to(13.365, 2.75).line_to(13.365, 3.5).line_to(
    15.165, 3.5
).line_to(15.165, 2.75).line_to(19.43, 2.75).line_to(19.43, 2.1).line_to(
    28.53, 2.1
).line_to(28.53, 0.0).line_to(0.0, 0.0)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, "X", "NewBodyFeatureOperation")

# Sketch 2
wp2 = app.work_plane("XY", offset=1.6)

wp2.move_to(0.65, 0.0).line_to(8.65, 0.0)
wp2.move_to(1.15, -0.5).line_to(8.15, -0.5).three_point_arc(
    (8.65, 0.0), (8.15, 0.5)
).line_to(1.15, 0.5).three_point_arc((0.65, 0.0), (1.15, -0.5))

# Extrude feature 2
shape2 = wp2.extrude(-0.5, "Cut", symmetric=False)

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(9.3, 1.8).line_to(9.32, 1.725359)
wp3.move_to(9.161962, 1.57).line_to(9.05, 1.6).line_to(9.05, 1.8).line_to(9.3, 1.8)
wp3.move_to(9.161962, 1.57).three_point_arc((9.295525, 1.592195), (9.32, 1.725359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, "X", "Cut")

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(0.0, 0.0).line_to(0.0, 0.5).line_to(0.095263, 0.335).line_to(
    0.402702, 0.1575
).line_to(1.009067, 0.1575).line_to(1.1, 0.0).line_to(0.0, 0.0)

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, "X", "Cut")

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(10.9, 2.45).line_to(10.92, 2.375359)
wp5.move_to(10.761962, 2.22).line_to(10.65, 2.25).line_to(10.65, 2.45).line_to(
    10.9, 2.45
)
wp5.move_to(10.761962, 2.22).three_point_arc((10.895525, 2.242195), (10.92, 2.375359))

# Revolve feature 5
shape5 = wp5.revolve(6.283185307179586, "X", "Cut")

# Sketch 6
wp6 = app.work_plane("XY")

wp6.move_to(11.35, 2.6).line_to(11.565, 2.6).line_to(11.565, 2.75).line_to(
    11.35, 2.75
).line_to(11.35, 2.6)

# Revolve feature 6
shape6 = wp6.revolve(6.283185307179586, "X", "Cut")

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(13.365, 2.95).line_to(13.385, 2.875359)
wp7.move_to(13.226962, 2.72).line_to(13.115, 2.75).line_to(13.115, 2.95).line_to(
    13.365, 2.95
)
wp7.move_to(13.226962, 2.72).three_point_arc((13.360525, 2.742195), (13.385, 2.875359))

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, "X", "Cut")

# Sketch 8
wp8 = app.work_plane("XY")

wp8.move_to(15.165, 2.95).line_to(15.145, 2.875359).three_point_arc(
    (15.169475, 2.742195), (15.303038, 2.72)
).line_to(15.415, 2.75).line_to(15.415, 2.95).line_to(15.165, 2.95)

# Revolve feature 8
shape8 = wp8.revolve(6.283185307179586, "X", "Cut")

# Sketch 9
wp9 = app.work_plane("XY", offset=2.75)

wp9.move_to(18.03, 0.0).line_to(19.83, 0.0)
wp9.move_to(18.43, -0.4).line_to(19.43, -0.4).three_point_arc(
    (19.83, 0.0), (19.43, 0.4)
).line_to(18.43, 0.4).three_point_arc((18.03, 0.0), (18.43, -0.4))

# Extrude feature 9
shape9 = wp9.extrude(-0.25, "Cut", symmetric=False)

# Sketch 10
wp10 = app.work_plane("XY", offset=2.1)

wp10.move_to(20.48, 0.0).line_to(27.48, 0.0)
wp10.move_to(21.08, -0.6).line_to(26.88, -0.6).three_point_arc(
    (27.48, 0.0), (26.88, 0.6)
).line_to(21.08, 0.6).three_point_arc((20.48, 0.0), (21.08, -0.6))

# Extrude feature 10
shape10 = wp10.extrude(-0.5, "Cut", symmetric=False)

# Sketch 11
wp11 = app.work_plane("XY")

wp11.move_to(19.43, 2.3).line_to(19.41, 2.225359).three_point_arc(
    (19.434475, 2.092195), (19.568038, 2.07)
).line_to(19.68, 2.1).line_to(19.68, 2.3).line_to(19.43, 2.3)

# Revolve feature 11
shape11 = wp11.revolve(6.283185307179586, "X", "Cut")
