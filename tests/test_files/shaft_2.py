from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 1.6).line_to(8.4, 1.6).line_to(8.4, 2.25).line_to(10.0, 2.25).line_to(10.0, 2.75).line_to(12.465, 2.75).line_to(12.465, 3.25).line_to(14.265, 3.25).line_to(14.265, 2.75).line_to(16.73, 2.75).line_to(16.73, 0.0).line_to(0.0, 0.0)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, 'X', 'NewBodyFeatureOperation')

# Sketch 2
wp2 = app.work_plane("XY", offset=1.6)

wp2.move_to(0.7, 0.0).line_to(7.7, 0.0)
wp2.move_to(1.2, -0.5).line_to(7.2, -0.5).three_point_arc((7.7, 0.0), (7.2, 0.5)).line_to(1.2, 0.5).three_point_arc((0.7, 0.0), (1.2, -0.5))

# Extrude feature 2
shape2 = wp2.extrude(-0.5, 'Cut', symmetric=False)

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(8.4, 1.8).line_to(8.42, 1.725359)
wp3.move_to(8.261962, 1.57).line_to(8.15, 1.6).line_to(8.15, 1.8).line_to(8.4, 1.8)
wp3.move_to(8.261962, 1.57).three_point_arc((8.395525, 1.592195), (8.42, 1.725359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(10.0, 2.45).line_to(10.02, 2.375359)
wp4.move_to(9.861962, 2.22).line_to(9.75, 2.25).line_to(9.75, 2.45).line_to(10.0, 2.45)
wp4.move_to(9.861962, 2.22).three_point_arc((9.995525, 2.242195), (10.02, 2.375359))

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(10.45, 2.6).line_to(10.665, 2.6).line_to(10.665, 2.75).line_to(10.45, 2.75).line_to(10.45, 2.6)

# Revolve feature 5
shape5 = wp5.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 6
wp6 = app.work_plane("XY")

wp6.move_to(12.465, 2.95).line_to(12.485, 2.875359)
wp6.move_to(12.326962, 2.72).line_to(12.215, 2.75).line_to(12.215, 2.95).line_to(12.465, 2.95)
wp6.move_to(12.326962, 2.72).three_point_arc((12.460525, 2.742195), (12.485, 2.875359))

# Revolve feature 6
shape6 = wp6.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(16.28, 2.6).line_to(16.065, 2.6).line_to(16.065, 2.75).line_to(16.28, 2.75).line_to(16.28, 2.6)

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 8
wp8 = app.work_plane("XY")

wp8.move_to(14.265, 2.95).line_to(14.245, 2.875359).three_point_arc((14.269475, 2.742195), (14.403038, 2.72)).line_to(14.515, 2.75).line_to(14.515, 2.95).line_to(14.265, 2.95)

# Revolve feature 8
shape8 = wp8.revolve(6.283185307179586, 'X', 'Cut')

# Chamfer feature 9
wp9.chamfer("X+", distance=0.17500000000000002, angle=1.8500490071139892)