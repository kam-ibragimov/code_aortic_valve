import re
from pathlib import Path

LANDMARK_KEYS = {'R', 'L', 'N', 'RLC', 'RNC', 'LNC'}
CURVE_KEYS = {'RGH', 'LGH', 'NGH', 'RCI', 'LCI', 'NCI', 'BR'}

_SPLINE_NAME_RE = re.compile(r'^Name:\s*(\S+)\s*$')
_SPLINE_POINT_RE = re.compile(r'^X\d+\s+Y\d+\s+Z\d+:\s*(.+)$')


def parse_slicer_txt(filepath):
    """Parses a Slicer-exported markup txt file (repeated Legend/Data sections,
    each containing a Line:, Point: or Spline: block).

    Returns a dict with only the structures the interobserver study needs:
        {'R': [x, y, z], ..., 'RGH': [[x, y, z], ...], ..., 'BR': [[x, y, z], ...]}

    Landmarks not in LANDMARK_KEYS and splines not in CURVE_KEYS (ST, RLS, RNS,
    LNS, ...) are dropped, and the Line: block (CC, RNV) is ignored entirely.
    """
    lines = Path(filepath).read_text().splitlines()

    result = {}
    in_data = False
    block = None  # 'point' | 'spline' | 'line' | None
    spline_name = None
    spline_points = []

    def flush_spline():
        nonlocal spline_name, spline_points
        if spline_name in CURVE_KEYS and spline_points:
            result[spline_name] = spline_points
        spline_name = None
        spline_points = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped == 'Legend':
            flush_spline()
            in_data = False
            block = None
            continue
        if stripped == 'Data':
            in_data = True
            block = None
            continue
        if not stripped or stripped.startswith('='):
            if not stripped:
                flush_spline()
                block = None
            continue
        if not in_data:
            continue

        if stripped in ('Point:', 'Spline:', 'Line:'):
            flush_spline()
            block = {'Point:': 'point', 'Spline:': 'spline', 'Line:': 'line'}[stripped]
            continue

        if block == 'point':
            if stripped.startswith('Name'):
                continue
            parts = stripped.split()
            if parts and parts[0] in LANDMARK_KEYS and len(parts) >= 4:
                result[parts[0]] = [float(parts[1]), float(parts[2]), float(parts[3])]
            continue

        if block == 'spline':
            name_match = _SPLINE_NAME_RE.match(stripped)
            if name_match:
                spline_name = name_match.group(1)
                spline_points = []
                continue
            point_match = _SPLINE_POINT_RE.match(stripped)
            if point_match:
                coords = point_match.group(1).split()
                if len(coords) >= 3:
                    spline_points.append([float(coords[0]), float(coords[1]), float(coords[2])])
            continue

        # block == 'line' (or None) — ignored on purpose

    flush_spline()
    return result


def get_case_number(filename):
    stem = Path(filename).stem
    m = re.search(r'\d+$', stem)
    return m.group() if m else None