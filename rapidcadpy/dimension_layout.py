"""Backend-neutral collision-aware dimension placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .drawing import DimensionIntent, DimensionPlacement
from .dimension_planner import ModelBounds

Point3D = tuple[float, float, float]
Point2D = tuple[float, float]
Rect = tuple[float, float, float, float]

@dataclass(frozen=True)
class ViewGeometry:
    name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    scale: float
    model_u_range: tuple[float, float]
    model_v_range: tuple[float, float]

    @property
    def rect(self) -> Rect:
        return (
            self.x_mm - self.width_mm / 2.0,
            self.y_mm - self.height_mm / 2.0,
            self.x_mm + self.width_mm / 2.0,
            self.y_mm + self.height_mm / 2.0,
        )

    def project(self, point: Point3D) -> Point2D:
        u, v = project_model_point(self.name, point)
        u_mid = (self.model_u_range[0] + self.model_u_range[1]) / 2.0
        v_mid = (self.model_v_range[0] + self.model_v_range[1]) / 2.0
        return (
            self.x_mm + (u - u_mid) * self.scale,
            self.y_mm + (v - v_mid) * self.scale,
        )


def place_dimension_intents(
    intents: Sequence[DimensionIntent],
    views: Sequence[ViewGeometry],
    *,
    standard: str,
    page_width_mm: float,
    page_height_mm: float,
    reserved_boxes: Sequence[Rect] = (),
    warnings: list[str] | None = None,
    allow_collisions: bool = False,
) -> tuple[DimensionPlacement, ...]:
    """Place dimension text outside views without collisions."""

    views_by_name = {view.name: view for view in views}
    occupied: list[Rect] = [*reserved_boxes]
    occupied_segments: list[tuple[float, float, float, float]] = []
    placements: list[DimensionPlacement] = []
    priority = {
        # Feature dimensions occupy the inner lanes; overall dimensions are
        # deliberately placed farther out, as required by drawing practice.
        "spacing": 0,
        "overall": 1,
        "hole": 2,
        "countersink": 2,
        "radius": 3,
        "chamfer": 4,
        "thickness": 5,
    }
    for intent in sorted(intents, key=lambda item: (priority[item.kind], item.id)):
        view = views_by_name.get(intent.view)
        if view is None:
            raise RuntimeError(
                f"Dimension {intent.id} references missing view {intent.view}."
            )
        projected = tuple(view.project(point) for point in intent.reference_points)
        label = intent.formatted_text(standard)
        width = max(10.0, len(label) * 2.15 + 3.0)
        height = 5.0
        placement = _find_placement(
            intent,
            projected,
            view,
            width=width,
            height=height,
            occupied=occupied,
            other_view_rects=[item.rect for item in views if item.name != intent.view],
            occupied_segments=occupied_segments,
            page_width_mm=page_width_mm,
            page_height_mm=page_height_mm,
        )
        if placement is None and allow_collisions:
            placement = _find_placement(
                intent, projected, view, width=width, height=height,
                occupied=occupied, other_view_rects=[item.rect for item in views if item.name != intent.view],
                occupied_segments=occupied_segments, page_width_mm=page_width_mm,
                page_height_mm=page_height_mm, allow_collisions=True,
            )
            if placement is not None and warnings is not None:
                warnings.append(
                    f"Dimension {intent.id} was placed with a collision warning."
                )
        if placement is None:
            if warnings is None:
                raise RuntimeError(
                    "The drawing page and views were created, but automatic dimension "
                    f"placement could not find a collision-free location for {intent.id}. "
                    "The existing drawing requires manual dimension placement."
                )
            warnings.append(
                "Automatic dimension placement could not find a collision-free "
                f"location for {intent.id}; that annotation was omitted."
            )
            continue
        placements.append(placement)
        occupied.append(placement.text_box)
        occupied_segments.extend(placement.leader_segments)
    if not allow_collisions:
        validate_dimension_placements(
            placements, views, page_width_mm=page_width_mm,
            page_height_mm=page_height_mm, reserved_boxes=reserved_boxes,
        )
    return tuple(placements)


def validate_dimension_placements(
    placements: Sequence[DimensionPlacement],
    views: Sequence[ViewGeometry],
    *,
    page_width_mm: float,
    page_height_mm: float,
    allow_collisions: bool = False,
    reserved_boxes: Sequence[Rect] = (),
) -> None:
    """Reject text overlap, sheet overflow, and dimension/view conflicts."""

    view_rects = [view.rect for view in views]
    boxes: list[Rect] = []
    for placement in placements:
        box = placement.text_box
        if (
            box[0] < 0.0
            or box[1] < 0.0
            or box[2] > page_width_mm
            or box[3] > page_height_mm
        ):
            raise RuntimeError(
                f"Dimension {placement.intent.id} falls outside the sheet."
            )
        if any(_rectangles_overlap(box, item, clearance=1.0) for item in view_rects):
            raise RuntimeError(
                f"Dimension {placement.intent.id} overlaps projected geometry."
            )
        if any(
            _rectangles_overlap(box, item, clearance=1.0) for item in reserved_boxes
        ):
            raise RuntimeError(
                f"Dimension {placement.intent.id} overlaps reserved sheet content."
            )
        if any(_rectangles_overlap(box, item, clearance=1.0) for item in boxes):
            raise RuntimeError(
                f"Dimension {placement.intent.id} overlaps another dimension."
            )
        boxes.append(box)

    for placement in placements:
        other_view_rects = [
            view.rect for view in views if view.name != placement.intent.view
        ]
        for segment in placement.leader_segments:
            if any(
                _segment_intersects_rect(segment, item) for item in other_view_rects
            ):
                raise RuntimeError(
                    f"Dimension {placement.intent.id} crosses another projected view."
                )
            for other in placements:
                if other is placement:
                    continue
                if _segment_intersects_rect(segment, other.text_box):
                    raise RuntimeError(
                        f"Dimension {placement.intent.id} crosses dimension "
                        f"{other.intent.id}."
                    )
    for index, first in enumerate(placements):
        for second in placements[index + 1 :]:
            if any(
                _segments_intersect(first_segment, second_segment)
                for first_segment in first.leader_segments
                for second_segment in second.leader_segments
            ):
                raise RuntimeError(
                    f"Dimensions {first.intent.id} and {second.intent.id} "
                    "have crossing lines."
                )


def project_model_point(view_name: str, point: Point3D) -> Point2D:
    if view_name == "front":
        return (point[0], point[2])
    if view_name == "top":
        return (point[0], point[1])
    if view_name == "right":
        return (point[1], point[2])
    raise ValueError(f"Unsupported dimension view {view_name!r}.")


def view_model_ranges(
    view_name: str, bounds: ModelBounds
) -> tuple[tuple[float, float], tuple[float, float]]:
    if view_name == "front":
        return (bounds.x_min, bounds.x_max), (bounds.z_min, bounds.z_max)
    if view_name == "top":
        return (bounds.x_min, bounds.x_max), (bounds.y_min, bounds.y_max)
    if view_name == "right":
        return (bounds.y_min, bounds.y_max), (bounds.z_min, bounds.z_max)
    raise ValueError(f"Unsupported dimension view {view_name!r}.")


def _find_placement(
    intent: DimensionIntent,
    projected: tuple[Point2D, ...],
    view: ViewGeometry,
    *,
    width: float,
    height: float,
    occupied: Sequence[Rect],
    other_view_rects: Sequence[Rect],
    occupied_segments: Sequence[tuple[float, float, float, float]],
    page_width_mm: float,
    page_height_mm: float,
    allow_collisions: bool = False,
) -> Optional[DimensionPlacement]:
    anchor_x = sum(point[0] for point in projected) / len(projected)
    anchor_y = sum(point[1] for point in projected) / len(projected)
    shifts = (0.0, 7.0, -7.0, 14.0, -14.0, 21.0, -21.0, 28.0, -28.0)
    offsets = (7.0, 13.0, 19.0, 25.0, 31.0)
    left, bottom, right, top = view.rect
    for side in intent.preferred_sides:
        for offset in offsets:
            for shift in shifts:
                if side == "right":
                    x = right + offset + width / 2.0
                    y = _clamp(
                        anchor_y + shift,
                        height / 2.0 + 2.0,
                        page_height_mm - height / 2.0 - 2.0,
                    )
                elif side == "left":
                    x = left - offset - width / 2.0
                    y = _clamp(
                        anchor_y + shift,
                        height / 2.0 + 2.0,
                        page_height_mm - height / 2.0 - 2.0,
                    )
                elif side == "top":
                    x = _clamp(
                        anchor_x + shift,
                        width / 2.0 + 2.0,
                        page_width_mm - width / 2.0 - 2.0,
                    )
                    y = top + offset + height / 2.0
                else:
                    x = _clamp(
                        anchor_x + shift,
                        width / 2.0 + 2.0,
                        page_width_mm - width / 2.0 - 2.0,
                    )
                    y = bottom - offset - height / 2.0
                box = (
                    x - width / 2.0,
                    y - height / 2.0,
                    x + width / 2.0,
                    y + height / 2.0,
                )
                if (
                    box[0] < 2.0
                    or box[1] < 2.0
                    or box[2] > page_width_mm - 2.0
                    or box[3] > page_height_mm - 2.0
                ):
                    continue
                if _rectangles_overlap(box, view.rect, clearance=1.0):
                    continue
                if not allow_collisions and any(
                    _rectangles_overlap(box, item, clearance=1.0)
                    for item in other_view_rects
                ):
                    continue
                if not allow_collisions and any(
                    _rectangles_overlap(box, item, clearance=1.0) for item in occupied
                ):
                    continue
                leader = _dimension_segments(intent, side, projected, x, y)
                if not allow_collisions and any(
                    _segment_intersects_rect(segment, item)
                    for segment in leader
                    for item in other_view_rects
                ):
                    continue
                if not allow_collisions and any(
                    _segment_intersects_rect(segment, item)
                    for segment in leader
                    for item in occupied
                ):
                    continue
                if not allow_collisions and any(
                    _segment_intersects_rect(existing, box)
                    for existing in occupied_segments
                ):
                    continue
                if any(
                    _segments_intersect(segment, existing)
                    for segment in leader
                    for existing in occupied_segments
                ):
                    continue
                return DimensionPlacement(
                    intent=intent,
                    side=side,
                    x_mm=x,
                    y_mm=y,
                    text_box=box,
                    projected_reference_points=projected,
                    leader_segments=leader,
                )
    return None


def _dimension_segments(
    intent: DimensionIntent,
    side: str,
    projected: tuple[Point2D, ...],
    x: float,
    y: float,
) -> tuple[tuple[float, float, float, float], ...]:
    if len(projected) >= 2 and intent.kind in {
        "overall",
        "spacing",
        "thickness",
    }:
        first, second = projected[0], projected[1]
        if side in {"top", "bottom"}:
            return (
                (first[0], first[1], first[0], y),
                (second[0], second[1], second[0], y),
                (first[0], y, second[0], y),
            )
        return (
            (first[0], first[1], x, first[1]),
            (second[0], second[1], x, second[1]),
            (x, first[1], x, second[1]),
        )
    anchor_x = sum(point[0] for point in projected) / len(projected)
    anchor_y = sum(point[1] for point in projected) / len(projected)
    return ((anchor_x, anchor_y, x, y),)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _rectangles_overlap(first: Rect, second: Rect, *, clearance: float = 0.0) -> bool:
    return not (
        first[2] + clearance <= second[0]
        or second[2] + clearance <= first[0]
        or first[3] + clearance <= second[1]
        or second[3] + clearance <= first[1]
    )


def _segment_intersects_rect(
    segment: tuple[float, float, float, float],
    rect: Rect,
) -> bool:
    first = (segment[0], segment[1])
    second = (segment[2], segment[3])
    if _point_in_rect(first, rect) or _point_in_rect(second, rect):
        return True
    left, bottom, right, top = rect
    edges = (
        (left, bottom, right, bottom),
        (right, bottom, right, top),
        (right, top, left, top),
        (left, top, left, bottom),
    )
    return any(_segments_cross(segment, edge) for edge in edges)


def _segments_intersect(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    first_points = ((first[0], first[1]), (first[2], first[3]))
    second_points = ((second[0], second[1]), (second[2], second[3]))
    if any(_points_close(a, b) for a in first_points for b in second_points):
        return False
    return _segments_cross(first, second)


def _segments_cross(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    p1 = (first[0], first[1])
    q1 = (first[2], first[3])
    p2 = (second[0], second[1])
    q2 = (second[2], second[3])
    orientations = (
        _orientation(p1, q1, p2),
        _orientation(p1, q1, q2),
        _orientation(p2, q2, p1),
        _orientation(p2, q2, q1),
    )
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True
    return (
        (orientations[0] == 0 and _point_on_segment(p1, p2, q1))
        or (orientations[1] == 0 and _point_on_segment(p1, q2, q1))
        or (orientations[2] == 0 and _point_on_segment(p2, p1, q2))
        or (orientations[3] == 0 and _point_on_segment(p2, q1, q2))
    )


def _orientation(first: Point2D, second: Point2D, third: Point2D) -> int:
    value = (second[1] - first[1]) * (third[0] - second[0]) - (second[0] - first[0]) * (
        third[1] - second[1]
    )
    if abs(value) <= 1.0e-8:
        return 0
    return 1 if value > 0.0 else 2


def _point_on_segment(first: Point2D, point: Point2D, second: Point2D) -> bool:
    return (
        min(first[0], second[0]) - 1.0e-8
        <= point[0]
        <= max(first[0], second[0]) + 1.0e-8
        and min(first[1], second[1]) - 1.0e-8
        <= point[1]
        <= max(first[1], second[1]) + 1.0e-8
    )


def _point_in_rect(point: Point2D, rect: Rect) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _points_close(first: Point2D, second: Point2D) -> bool:
    return abs(first[0] - second[0]) <= 1.0e-8 and abs(first[1] - second[1]) <= 1.0e-8


__all__ = [
    "ViewGeometry",
    "place_dimension_intents",
    "project_model_point",
    "validate_dimension_placements",
    "view_model_ranges",
]
