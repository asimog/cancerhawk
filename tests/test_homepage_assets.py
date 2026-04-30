"""Regression tests for homepage brand assets and pipeline wiring."""

from __future__ import annotations

import ast
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _png_rgba_pixels(path: Path) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")

    pos = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    assert width and height
    assert bit_depth == 8
    assert color_type == 6

    raw = zlib.decompress(bytes(compressed))
    stride = width * 4
    rows: list[bytearray] = []
    offset = 0

    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        prior = rows[-1] if rows else bytearray(stride)

        for i in range(stride):
            left = row[i - 4] if i >= 4 else 0
            up = prior[i]
            up_left = prior[i - 4] if i >= 4 else 0
            if filter_type == 1:
                row[i] = (row[i] + left) & 0xFF
            elif filter_type == 2:
                row[i] = (row[i] + up) & 0xFF
            elif filter_type == 3:
                row[i] = (row[i] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                row[i] = (row[i] + predictor) & 0xFF
            else:
                assert filter_type == 0
        rows.append(row)

    pixels = [
        tuple(row[i : i + 4])
        for row in rows
        for i in range(0, len(row), 4)
    ]
    return width, height, pixels


def test_homepage_logo_png_has_real_transparency():
    width, height, pixels = _png_rgba_pixels(ROOT / "public/logo.png")
    alphas = [pixel[3] for pixel in pixels]
    corners = [
        pixels[0],
        pixels[width - 1],
        pixels[(height - 1) * width],
        pixels[(height * width) - 1],
    ]

    assert min(alphas) == 0
    assert max(alphas) == 255
    assert sum(alpha <= 16 for alpha in alphas) > width * height * 0.35
    assert all(alpha == 0 for *_, alpha in corners)


def test_home_logo_style_does_not_add_a_background():
    css = (ROOT / "src/styles/globals.css").read_text()
    home_logo_rule = css.split(".home-logo", 1)[1].split("}", 1)[0]

    assert "background" not in home_logo_rule
    assert "box-shadow" not in home_logo_rule


def test_miroshark_peer_review_remains_after_paper_and_before_simulations():
    source = (ROOT / "app/hermes_supervisor.py").read_text()
    module = ast.parse(source)
    run_method = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    calls: list[str] = []

    class OrderedCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.append(fn.attr)
            self.generic_visit(node)

    OrderedCallVisitor().visit(run_method)

    assert calls.index("run_paper_engine") < calls.index("run_peer_review_engine")
    assert calls.index("run_peer_review_engine") < calls.index("generate_html5_simulations")
    assert "reviews_to_dict" in calls
    assert "consolidated_to_dict" in calls
