from rapidcadpy.dimension_layout import ViewGeometry, place_dimension_intents
from rapidcadpy.dimension_planner import ModelBounds, build_dimension_intents_from_features
from rapidcadpy.features import HoleFeature, HoleTermination


def test_dimension_layout_places_semantic_dimensions_outside_views() -> None:
    holes = tuple(
        HoleFeature(
            target_id="body",
            center=(x, y, 5.0),
            diameter_mm=10.0,
            termination=HoleTermination.THROUGH,
        )
        for x, y in ((20.0, 15.0), (60.0, 15.0), (20.0, 35.0), (60.0, 35.0))
    )
    intents = build_dimension_intents_from_features(
        ModelBounds(0.0, 80.0, 0.0, 50.0, 0.0, 10.0), holes
    )
    views = (
        ViewGeometry("front", 170.0, 170.0, 80.0, 10.0, 1.0, (0.0, 80.0), (0.0, 10.0)),
        ViewGeometry("top", 170.0, 65.0, 80.0, 50.0, 1.0, (0.0, 80.0), (0.0, 50.0)),
        ViewGeometry("right", 60.0, 170.0, 50.0, 10.0, 1.0, (0.0, 50.0), (0.0, 10.0)),
        ViewGeometry("isometric", 315.0, 175.0, 90.0, 70.0, 1.0, (0.0, 1.0), (0.0, 1.0)),
    )

    placements = place_dimension_intents(
        intents,
        views,
        standard="ISO",
        page_width_mm=420.0,
        page_height_mm=297.0,
        reserved_boxes=((250.0, 0.0, 420.0, 60.0),),
    )

    assert len(placements) == len(intents)
    for placement in placements:
        assert all(
            placement.text_box[2] <= view.rect[0]
            or view.rect[2] <= placement.text_box[0]
            or placement.text_box[3] <= view.rect[1]
            or view.rect[3] <= placement.text_box[1]
            for view in views
        )
