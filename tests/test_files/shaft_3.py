from rapidcadpy import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 2.5).line_to(2.265, 2.5).line_to(2.265, 3.15).line_to(4.065, 3.15).line_to(4.065, 2.5).line_to(6.33, 2.5).line_to(6.33, 2.1).line_to(7.93, 2.1).line_to(7.93, 1.3).line_to(14.13, 1.3).line_to(14.13, 0.0).line_to(0.0, 0.0)

# Revolve feature 1
shape1 = wp1.revolve(6.283185307179586, 'X', 'NewBodyFeatureOperation')

# Sketch 2
wp2 = app.work_plane("XY")

wp2.move_to(0.45, 2.35).line_to(0.665, 2.35).line_to(0.665, 2.5).line_to(0.45, 2.5).line_to(0.45, 2.35)

# Revolve feature 2
shape2 = wp2.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(2.265, 2.7).line_to(2.285, 2.625359)
wp3.move_to(2.126962, 2.47).line_to(2.015, 2.5).line_to(2.015, 2.7).line_to(2.265, 2.7)
wp3.move_to(2.126962, 2.47).three_point_arc((2.260525, 2.492195), (2.285, 2.625359))

# Revolve feature 3
shape3 = wp3.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(0.0, 0.0).line_to(0.0, 0.8).line_to(0.155885, 0.53).line_to(0.640859, 0.25).line_to(1.055662, 0.25).line_to(1.2, 0.0).line_to(0.0, 0.0)

# Revolve feature 4
shape4 = wp4.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(5.88, 2.35).line_to(5.665, 2.35).line_to(5.665, 2.5).line_to(5.88, 2.5).line_to(5.88, 2.35)

# Revolve feature 5
shape5 = wp5.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 6
wp6 = app.work_plane("XY")

wp6.move_to(4.065, 2.7).line_to(4.045, 2.625359).three_point_arc((4.069475, 2.492195), (4.203038, 2.47)).line_to(4.315, 2.5).line_to(4.315, 2.7).line_to(4.065, 2.7)

# Revolve feature 6
shape6 = wp6.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(6.33, 2.3).line_to(6.31, 2.225359).three_point_arc((6.334475, 2.092195), (6.468038, 2.07)).line_to(6.58, 2.1).line_to(6.58, 2.3).line_to(6.33, 2.3)

# Revolve feature 7
shape7 = wp7.revolve(6.283185307179586, 'X', 'Cut')

# Sketch 8
wp8 = app.work_plane("XY", offset=1.3)

wp8.move_to(8.53, 0.0).line_to(13.53, 0.0)
wp8.move_to(8.93, -0.4).line_to(13.13, -0.4).three_point_arc((13.53, 0.0), (13.13, 0.4)).line_to(8.93, 0.4).three_point_arc((8.53, 0.0), (8.93, -0.4))

# Extrude feature 8
shape8 = wp8.extrude(-0.4, 'Cut', symmetric=False)

# Sketch 9
wp9 = app.work_plane("XY")

wp9.move_to(7.93, 1.5).line_to(7.91, 1.425359).three_point_arc((7.934475, 1.292195), (8.068038, 1.27)).line_to(8.18, 1.3).line_to(8.18, 1.5).line_to(7.93, 1.5)

# Revolve feature 9
shape9 = wp9.revolve(6.283185307179586, 'X', 'Cut')
