"""Regression tests for OMML run-style fidelity (m:rPr/m:scr, m:sty).

P3-h: parse_r previously dropped m:rPr entirely, so styled math variables
(double-struck, fraktur, script, bold) rendered as plain symbols. These tests
assert the LaTeX macro wrapping the run's content.
"""

from lxml import etree

from lightrag.parser.docx.omml.ommlparser import OMMLParser

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _run_xml(inner_rpr: str, text: str) -> bytes:
    return f"""
    <m:oMath xmlns:m="{M_NS}">
      <m:r>
        {inner_rpr}
        <m:t>{text}</m:t>
      </m:r>
    </m:oMath>
    """.encode()


def _parse(inner_rpr: str, text: str = "R") -> str:
    root = etree.fromstring(_run_xml(inner_rpr, text))
    return OMMLParser().parse(root)


def test_scr_double_struck_wraps_mathbb():
    rpr = f'<m:rPr><m:scr m:val="double-struck" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr) == "\\mathbb{R}"


def test_scr_fraktur_wraps_mathfrak():
    rpr = f'<m:rPr><m:scr m:val="fraktur" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "g") == "\\mathfrak{g}"


def test_scr_script_wraps_mathcal():
    rpr = f'<m:rPr><m:scr m:val="script" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "L") == "\\mathcal{L}"


def test_scr_sans_serif_wraps_mathsf():
    rpr = f'<m:rPr><m:scr m:val="sans-serif" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "x") == "\\mathsf{x}"


def test_scr_monospace_wraps_mathtt():
    rpr = f'<m:rPr><m:scr m:val="monospace" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "x") == "\\mathtt{x}"


def test_sty_bold_wraps_mathbf():
    rpr = f'<m:rPr><m:sty m:val="b" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "v") == "\\mathbf{v}"


def test_sty_bold_italic_wraps_boldsymbol():
    rpr = f'<m:rPr><m:sty m:val="bi" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr, "v") == "\\boldsymbol{v}"


def test_scr_takes_precedence_over_sty():
    rpr = (
        f'<m:rPr><m:sty m:val="b" xmlns:m="{M_NS}"/>'
        f'<m:scr m:val="fraktur" xmlns:m="{M_NS}"/></m:rPr>'
    )
    assert _parse(rpr, "g") == "\\mathfrak{g}"


def test_unknown_scr_value_passes_through_unwrapped():
    rpr = f'<m:rPr><m:scr m:val="bogus" xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr) == "R"


def test_no_rpr_passes_through_unwrapped():
    assert _parse("") == "R"


def test_rpr_without_scr_or_sty_passes_through_unwrapped():
    rpr = f'<m:rPr><m:nor xmlns:m="{M_NS}"/></m:rPr>'
    assert _parse(rpr) == "R"
