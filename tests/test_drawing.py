from __future__ import annotations

import sys

import pytest

from rapidcadpy.drawing import (
    _validate_uncompressed_pdf,
    finalize_vector_pdf,
    validate_print_ready_pdf,
)


def test_finalize_vector_pdf_preserves_a3_and_embeds_fonts(tmp_path, monkeypatch):
    pytest.importorskip("weasyprint")
    pytest.importorskip("pypdf")
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="420mm" height="297mm" viewBox="0 0 420 297">'
            '<rect x="5" y="5" width="410" height="287" '
            'fill="none" stroke="black"/>'
            '<text x="20" y="20" font-family="DejaVu Sans" '
            'font-size="4">Technical drawing</text></svg>'
        ),
        encoding="utf-8",
    )
    pdf = tmp_path / "drawing.pdf"

    validation = finalize_vector_pdf(svg_path=svg, pdf_path=pdf)

    assert pdf.is_file()
    assert validation["page_width_mm"] == pytest.approx(420.0, abs=1.0)
    assert validation["page_height_mm"] == pytest.approx(297.0, abs=1.0)
    assert validation["fonts_embedded"] is True

    fallback_validation = _validate_uncompressed_pdf(
        pdf,
        page_width_mm=420.0,
        page_height_mm=297.0,
    )

    assert fallback_validation["fonts_embedded"] is True
    assert fallback_validation["validator"] == "builtin"

    monkeypatch.setitem(sys.modules, "pypdf", None)
    public_fallback = validate_print_ready_pdf(pdf)

    assert public_fallback["fonts_embedded"] is True
    assert public_fallback["validator"] == "builtin"
