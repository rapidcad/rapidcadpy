from rapidcadpy import OpenCascadeOcpApp

# Initialize Inventor application
app = OpenCascadeOcpApp()
app.new_document()

# Sketch 1
wp1 = app.work_plane("XY")

wp1.move_to(0.0, 0.0).line_to(0.0, 2.6).line_to(6.5, 2.6).line_to(6.5, 3.5).line_to(
    8.8, 3.5
).line_to(8.8, 2.75).line_to(13.065, 2.75).line_to(13.065, 2.05).line_to(
    25.565, 2.05
).line_to(
    25.565, 1.5
).line_to(
    27.235, 1.5
).line_to(
    27.235, 1.25
).line_to(
    28.635, 1.25
).line_to(
    28.635, 0.6
).line_to(
    35.135, 0.6
).line_to(
    35.135, 0.0
).line_to(
    0.0, 0.0
)

# Revolve feature 1
shape1 = wp1.revolve(1.0, "X", "NewBodyFeatureOperation")

# Sketch 2
wp2 = app.work_plane("XY", offset=2.6)

# wp2.move_to(1.0, 0.0).line_to(5.5, 0.0)
wp2.move_to(1.8, -0.8).line_to(4.7, -0.8).three_point_arc(
    (5.5, 0.0), (4.7, 0.8)
).line_to(1.8, 0.8).three_point_arc((1.0, 0.0), (1.8, -0.8))

# Extrude feature 2
shape2 = wp2.extrude(-0.6, "Cut", symmetric=False)

# Sketch 3
wp3 = app.work_plane("XY")

wp3.move_to(6.5, 2.8).line_to(6.5195, 2.6611)
wp3.move_to(6.4403, 2.57).line_to(6.3725, 2.57)
wp3.move_to(6.3518, 2.5727).line_to(6.25, 2.6).line_to(6.25, 2.8).line_to(6.5, 2.8)
wp3.move_to(6.4403, 2.57).three_point_arc((6.5007, 2.5975), (6.5195, 2.6611))
wp3.move_to(6.3518, 2.5727).three_point_arc((6.3621, 2.5707), (6.3725, 2.57))

# Revolve feature 3
shape3 = wp3.revolve(1.0, "X", "Cut")

# Sketch 4
wp4 = app.work_plane("XY")

wp4.move_to(0.45, 2.45).line_to(0.665, 2.45).line_to(0.665, 2.6).line_to(
    0.45, 2.6
).line_to(0.45, 2.45)

# Revolve feature 4
shape4 = wp4.revolve(1.0, "X", "Cut")

# Sketch 5
wp5 = app.work_plane("XY")

wp5.move_to(8.8, 2.95).line_to(8.7805, 2.8111).three_point_arc(
    (8.7993, 2.7475), (8.8597, 2.72)
).line_to(8.9275, 2.72).three_point_arc((8.9379, 2.7207), (8.9482, 2.7227)).line_to(
    9.05, 2.75
).line_to(
    9.05, 2.95
).line_to(
    8.8, 2.95
)

# Revolve feature 5
shape5 = wp5.revolve(1.0, "X", "Cut")

# Sketch 6
wp6 = app.work_plane("XY", offset=2.75)

# wp6.move_to(11.615, 0.0).line_to(13.515, 0.0)
wp6.move_to(12.065, -0.45).line_to(13.065, -0.45).three_point_arc(
    (13.515, 0.0), (13.065, 0.45)
).line_to(12.065, 0.45).three_point_arc((11.615, 0.0), (12.065, -0.45))

# Extrude feature 6
shape6 = wp6.extrude(-0.4, "Cut", symmetric=False)

# Sketch 7
wp7 = app.work_plane("XY")

wp7.move_to(27.025, 1.43).line_to(26.865, 1.43).line_to(26.865, 1.5).line_to(
    27.025, 1.5
).line_to(27.025, 1.43)

# Revolve feature 7
shape7 = wp7.revolve(1.0, "X", "Cut")

# Sketch 8
wp8 = app.work_plane("XY")

wp8.move_to(25.565, 1.7).line_to(25.5455, 1.5611).three_point_arc(
    (25.5643, 1.4975), (25.6247, 1.47)
).line_to(25.6925, 1.47).three_point_arc((25.7029, 1.4707), (25.7132, 1.4727)).line_to(
    25.815, 1.5
).line_to(
    25.815, 1.7
).line_to(
    25.565, 1.7
)

# Revolve feature 8
shape8 = wp8.revolve(1.0, "X", "Cut")

# Sketch 9
wp9 = app.work_plane("XY")

wp9.move_to(27.235, 1.45).line_to(27.2155, 1.3111).three_point_arc(
    (27.2343, 1.2475), (27.2947, 1.22)
).line_to(27.3625, 1.22).three_point_arc((27.3729, 1.2207), (27.3832, 1.2227)).line_to(
    27.485, 1.25
).line_to(
    27.485, 1.45
).line_to(
    27.235, 1.45
)

# Revolve feature 9
shape9 = wp9.revolve(1.0, "X", "Cut")

# Sketch 10
wp10 = app.work_plane("XY", offset=0.6)

# wp10.move_to(29.635, 0.0).line_to(34.135, 0.0)
wp10.move_to(29.835, -0.2).line_to(33.935, -0.2).three_point_arc(
    (34.135, 0.0), (33.935, 0.2)
).line_to(29.835, 0.2).three_point_arc((29.635, 0.0), (29.835, -0.2))

# Extrude feature 10
shape10 = wp10.extrude(-0.25, "Cut", symmetric=False)

# Sketch 11
wp11 = app.work_plane("XY")

wp11.move_to(28.635, 0.69).line_to(28.6259, 0.6256).three_point_arc(
    (28.6354, 0.5938), (28.6656, 0.58)
).line_to(28.7551, 0.58).three_point_arc((28.7603, 0.5803), (28.7654, 0.5814)).line_to(
    28.835, 0.6
).line_to(
    28.835, 0.69
).line_to(
    28.635, 0.69
)

# Revolve feature 11
shape11 = wp11.revolve(1.0, "X", "Cut")

# Sketch 12
wp12 = app.work_plane("XY")

wp12.move_to(35.055, 0.575).line_to(34.945, 0.575).line_to(34.945, 0.6).line_to(
    35.055, 0.6
).line_to(35.055, 0.575)

# Revolve feature 12
shape12 = wp12.revolve(1.0, "X", "Cut")

# Chamfered Edges
app.chamfer_edge(x=35.055, radius=0.6, angle=2.3562, distance=0.1)
app.chamfer_edge(x=35.135, radius=0.6, angle=2.3562, distance=0.1)
app.chamfer_edge(x=28.635, radius=1.25, angle=1.9548, distance=0.125)
app.chamfer_edge(x=0.0, radius=2.6, angle=2.3562, distance=0.1)

# === Thread Features ===
# Thread: M55x2 (external)
thread = app.add_thread(
    x=10.9325,  # Cylindrical face center X
    radius=2.75,  # Cylindrical face radius
    axis="X",  # Cylinder axis direction
    designation="M55x1",
    thread_class="6H",
    thread_type="external",
    right_handed=True,
    modeled=True,
    length=2.1,
)


app.show_3d(screenshot="shaft_with_thread.png")
app.to_stl("shaft_with_thread.stl")
