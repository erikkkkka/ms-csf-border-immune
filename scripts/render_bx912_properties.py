"""Render the BX-912 structure and properties while retaining other PDF panels."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import pymupdf as fitz
import numpy as np
import pypdfium2 as pdfium
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


ROOT = Path(__file__).resolve().parents[1]
REGION = fitz.Rect(296, 409, 518.74, 528)
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def rich_text(c, x, y, runs, *, font="Arial", size=6, right=False):
    widths = [
        pdfmetrics.stringWidth(text, font, size * scale)
        for text, scale, shift in runs
    ]
    cursor = x - sum(widths) if right else x
    for (text, scale, shift), width in zip(runs, widths):
        c.setFont(font, size * scale)
        c.drawString(cursor, y + shift, text)
        cursor += width
    return sum(widths)


def structure_svg(molecule, destination):
    mol = Chem.MolFromSmiles(molecule["molecule_structures"]["canonical_smiles"])
    if mol is None:
        raise ValueError("Invalid reference SMILES")
    assert Chem.rdMolDescriptors.CalcMolFormula(mol) == "C20H23BrN8O"
    assert abs(Descriptors.MolWt(mol) - 471.36) < 0.01
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(480, 410)
    options = drawer.drawOptions()
    options.rotate = 90
    options.padding = 0.04
    options.bondLineWidth = 2.8
    options.fixedFontSize = 27
    options.fontFile = str(FONT_DIR / "Arial.ttf")
    options.useBWAtomPalette()
    options.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    destination.write_text(drawer.GetDrawingText(), encoding="utf-8")
    return mol


def panel_overlay(width, height, svg_path, values):
    pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("ArialBold", str(FONT_DIR / "Arial Bold.ttf")))
    stream = io.BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    c.setFillColorRGB(0.08, 0.08, 0.08)
    c.setFont("ArialBold", 8)
    c.drawString(300, height - 421.2, "g")
    c.drawString(310, height - 421.2, "BX-912 properties")

    drawing = svg2rlg(str(svg_path))
    scale = min(91 / drawing.width, 80 / drawing.height)
    original_width, original_height = drawing.width, drawing.height
    drawing.scale(scale, scale)
    drawing.width, drawing.height = original_width * scale, original_height * scale
    left = 300 + (91 - drawing.width) / 2
    bottom = height - 512 + (80 - drawing.height) / 2
    renderPDF.draw(drawing, c, left, bottom)

    label_x, value_right = 402, 508
    body_size = 5.8
    rows = [
        (446, [("MW", 1, 0)], [(f"{values['Molecular weight']:.2f} Da", 1, 0)]),
        (460, [("tPSA", 1, 0)], [(f"{values['Polar surface area']:.2f} \u00c5", 1, 0), ("2", 0.72, 2.2)]),
        (482, [("PAMPA P", 1, 0), ("eff", 0.72, -1.1)], [
            (f"{values['PAMPA Peff']:.2f} \u00d7 10", 1, 0),
            ("\u22127", 0.72, 2.2), (" cm s", 1, 0), ("\u22121", 0.72, 2.2),
        ]),
        (500, [("HepG2 IC", 1, 0), ("50", 0.72, -1.1)], [
            (f"{values['HepG2 viability IC50']:.1f} \u03bcM", 1, 0),
        ]),
    ]
    for baseline, label, value in rows:
        y = height - baseline
        c.setFillColorRGB(0.28, 0.30, 0.32)
        label_width = rich_text(c, label_x, y, label, size=body_size)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        value_width = rich_text(
            c, value_right, y, value, font="ArialBold", size=body_size, right=True,
        )
        assert value_right - value_width - label_x - label_width > 5
    c.setStrokeColorRGB(0.78, 0.80, 0.81)
    c.setLineWidth(0.35)
    c.line(label_x, height - 470, value_right, height - 470)
    c.save()
    return stream.getvalue()


def render(path, output, scale=4):
    document = pdfium.PdfDocument(str(path))
    image = document[0].render(scale=scale).to_pil().convert("RGB")
    image.save(output)
    return np.asarray(image)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    output_dir = args.output_dir or ROOT / "reproduced_figures/panel5g"
    output_dir.mkdir(parents=True, exist_ok=True)
    molecule = json.loads((ROOT / "data/reference/chembl_bx912_molecule.json").read_text())
    with (ROOT / "figures_main/source_data/Fig5g_developability_gate.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = {row["metric"]: float(row["numeric_value"]) for row in rows if row["numeric_value"]}
    assert values == {
        "Molecular weight": 471.36, "Polar surface area": 110.86,
        "PAMPA Peff": 1.39, "HepG2 viability IC50": 3.3,
    }
    svg_path = output_dir / "BX912_structure.svg"
    structure_svg(molecule, svg_path)
    document = fitz.open(args.input)
    assert len(document) == 1
    page = document[0]
    width, height = page.rect.width, page.rect.height
    original_words = [word[:5] for word in page.get_text("words") if not fitz.Rect(word[:4]).intersects(REGION)]
    page.add_redact_annot(REGION, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0, text=0)
    overlay = fitz.open(stream=panel_overlay(width, height, svg_path, values), filetype="pdf")
    page.show_pdf_page(page.rect, overlay, 0)
    output = output_dir / "Figure_5_transition_aware_chemical_probe.pdf"
    document.save(output, garbage=4, deflate=True)
    document.close()
    result = fitz.open(output)
    revised_words = [word[:5] for word in result[0].get_text("words") if not fitz.Rect(word[:4]).intersects(REGION)]
    assert original_words == revised_words, "Text outside panel g changed"
    panel_text = result[0].get_text("text", clip=REGION)
    for expected in ["471.36", "110.86", "1.39", "3.3", "\u22127", "\u22121", "\u03bcM"]:
        assert expected in panel_text, f"Missing glyph or value: {expected}"
    before = render(args.input, output_dir / "before.png")
    after = render(output, output_dir / "after.png")
    outside = np.ones(before.shape[:2], dtype=bool)
    outside[int(REGION.y0 * 4) - 2:int(REGION.y1 * 4) + 2,
            int(REGION.x0 * 4) - 2:int(REGION.x1 * 4) + 2] = False
    changed = np.any(before != after, axis=2)
    outside_changes = int(np.sum(changed & outside))
    assert outside_changes == 0, f"Changed {outside_changes} pixels outside panel g"
    image = pdfium.PdfDocument(str(output))[0].render(scale=6).to_pil()
    image.crop(tuple(round(value * 6) for value in REGION)).save(output_dir / "panel_g.png")
    qa = {
        "input": str(args.input.resolve()), "output": str(output.resolve()),
        "changed_region_points": list(REGION), "outside_changed_pixels": outside_changes,
        "source_chembl_id": molecule["molecule_chembl_id"], "source_values": values,
        "vector_structure": True, "outside_text_unchanged": True,
        "panel_text": panel_text,
    }
    (output_dir / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
