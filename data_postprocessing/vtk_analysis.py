from collections import deque
from pathlib import Path
from skimage.morphology import skeletonize
import SimpleITK as sitk
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import convolve as nd_convolve, label as nd_label, generate_binary_structure
from data_postprocessing.mesh_utils import vtk_meshing, np2sitk

import vtk


# ------------------
# VTK helpers
# ------------------

def _vtk_polyline_from_points(points):
    """Creates a polygonal chain connecting all points in order."""
    vtk_points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()

    polyline.GetPointIds().SetNumberOfIds(len(points))

    for i, p in enumerate(points):
        pid = vtk_points.InsertNextPoint(p)
        polyline.GetPointIds().SetId(i, pid)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polyline = vtk.vtkPolyData()
    polyline.SetPoints(vtk_points)
    polyline.SetLines(cells)
    return polyline


def _vtk_polyline_to_linesegments(polydata):
    """Create a set of segments from the original polygonal chain."""
    pts = polydata.GetPoints()
    n = pts.GetNumberOfPoints()

    lines = vtk.vtkCellArray()

    for i in range(n - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    out = vtk.vtkPolyData()
    out.SetPoints(pts)
    out.SetLines(lines)
    return out


def _vtk_clip_polyline_with_mesh(polyline, tub_mesh):
    """Trims the polygonal chain so that only the part lying inside the mesh remains."""

    # Create an implicit function from the closed mesh
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(tub_mesh)

    # Clip polyline with implicit function
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polyline)
    clipper.SetClipFunction(implicit)
    clipper.InsideOutOn()  # Keep inside
    clipper.Update()

    clipped_polyline = clipper.GetOutput()
    return clipped_polyline


# ------------------
# Main class
# ------------------

class CenterlineExtractor:

    def __init__(self, poly_degree=8, dense_samples=200, spline_smoothing=1.0,
                 max_gap_voxels=50,
                 min_large_fraction=0.15,
                 min_large_voxels=5,
                 chain_small_fragments=False,
                 max_small_gap_voxels=25):
        self.poly_degree = poly_degree
        self.dense_samples = dense_samples
        # spline_smoothing * N controls the total residual budget for splprep.
        # 1.0 = mild smoothing (removes staircase, keeps local bends).
        # Increase to 3-5 for heavier smoothing; set to 0 for exact interpolation.
        self.spline_smoothing = spline_smoothing
        # Component chaining parameters.
        # max_gap_voxels: max voxel distance between endpoints to chain large components.
        #   At 0.4 mm/vox → 50 vox = 20 mm.
        self.max_gap_voxels = max_gap_voxels
        # A component is "large" (= real structure) when its path length satisfies
        # BOTH thresholds simultaneously.
        self.min_large_voxels = min_large_voxels        # absolute minimum path length
        self.min_large_fraction = min_large_fraction    # fraction of the longest component
        # chain_small_fragments: if True, also attach "small" (noise) components to the
        # chain endpoints when they are within max_small_gap_voxels.  Off by default —
        # enable to test whether small peripheral fragments belong to the trajectory.
        self.chain_small_fragments = chain_small_fragments
        self.max_small_gap_voxels = max_small_gap_voxels

    # ---- Step 1: load and skeletonize ----
    def _load_and_skeleton(self, arr, nii):
        if isinstance(nii, (Path, str)):
            img = sitk.ReadImage(str(nii))
        else:
            img = nii
        arr = arr.astype(bool)

        # Crop to bounding box before skeletonize: avoids processing the full
        # volume when the structure occupies only a small region.
        foreground = np.argwhere(arr)
        skeleton_arr = np.zeros_like(arr, dtype=bool)
        if len(foreground) > 0:
            lo = np.maximum(foreground.min(axis=0) - 1, 0)
            hi = np.minimum(foreground.max(axis=0) + 2, np.array(arr.shape))
            slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
            skeleton_arr[slices] = skeletonize(arr[slices])

        skeleton_sitk = np2sitk(skeleton_arr.astype(np.uint8), img.GetSpacing())
        skeleton_sitk.CopyInformation(img)
        return img, skeleton_arr, skeleton_sitk

    # ---- Step 2: collect skeleton points ----
    def _order_skeleton_points(self, skeleton_arr):
        """Order skeleton voxels into a continuous path, chaining disconnected components.

        Each connected component is ordered internally via double-BFS (tree diameter).
        Components are then classified as "large" (real structure) or "small" (noise),
        and large components are chained greedily by nearest endpoint pairs within
        max_gap_voxels.  Small fragment chaining is controlled by chain_small_fragments.
        """
        raw_points = np.argwhere(skeleton_arr > 0)
        if len(raw_points) <= 1:
            return raw_points

        # ---- helpers ----
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        kernel[1, 1, 1] = 0
        neighbor_count = nd_convolve(
            skeleton_arr.astype(np.uint8), kernel, mode='constant', cval=0
        )
        struct26 = generate_binary_structure(3, 3)
        comp_labeled, ncomp = nd_label(skeleton_arr, structure=struct26)

        def neighbors_26_in(pt, vset):
            x, y, z = pt
            return [(x+dx, y+dy, z+dz)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    if (dx, dy, dz) != (0, 0, 0)
                    and (x+dx, y+dy, z+dz) in vset]

        def longest_path_in(comp_mask):
            """Double-BFS diameter path within one connected component."""
            vset = {tuple(p) for p in np.argwhere(comp_mask)}

            def bfs(start):
                parents = {start: None}
                q = deque([(start, 0)])
                far, md = start, 0
                while q:
                    node, d = q.popleft()
                    if d > md:
                        md, far = d, node
                    for nb in neighbors_26_in(node, vset):
                        if nb not in parents:
                            parents[nb] = node
                            q.append((nb, d + 1))
                return far, parents

            # Seed: degree-1 endpoint > any connected voxel > any voxel
            eps = np.argwhere(comp_mask & (neighbor_count == 1))
            conn = np.argwhere(comp_mask & (neighbor_count >= 1))
            any_v = np.argwhere(comp_mask)
            if len(eps) > 0:
                seed = tuple(eps[0])
            elif len(conn) > 0:
                seed = tuple(conn[0])
            else:
                seed = tuple(any_v[0])

            u, _ = bfs(seed)
            v, parents = bfs(u)
            path = []
            node = v
            while node is not None:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path

        # ---- order each component ----
        components = []  # list of {'path': [...], 'size': int, 'is_large': bool}
        for ci in range(1, ncomp + 1):
            path = longest_path_in(comp_labeled == ci)
            components.append({'path': path, 'size': len(path)})

        max_size = max(c['size'] for c in components)
        for c in components:
            c['is_large'] = (c['size'] >= self.min_large_voxels and
                             c['size'] >= self.min_large_fraction * max_size)

        components.sort(key=lambda c: c['size'], reverse=True)

        # ---- greedy chaining ----
        def greedy_attach(chain, candidates, max_gap):
            """Repeatedly attach the nearest candidate endpoint to either chain end."""
            candidates = list(candidates)  # work on a copy
            changed = True
            while changed and candidates:
                changed = False
                best = None  # (gap, idx, append_to_end, reverse_path)
                cs = np.array(chain[0], dtype=float)
                ce = np.array(chain[-1], dtype=float)
                for idx, cd in enumerate(candidates):
                    p0 = np.array(cd['path'][0], dtype=float)
                    pn = np.array(cd['path'][-1], dtype=float)
                    for gap, app_end, rev in [
                        (np.linalg.norm(ce - p0), True,  False),  # chain_end → comp_start
                        (np.linalg.norm(ce - pn), True,  True),   # chain_end → comp_end
                        (np.linalg.norm(cs - p0), False, True),   # chain_start ← comp_start
                        (np.linalg.norm(cs - pn), False, False),  # chain_start ← comp_end
                    ]:
                        if gap <= max_gap and (best is None or gap < best[0]):
                            best = (gap, idx, app_end, rev)
                if best is not None:
                    _, idx, app_end, rev = best
                    cd = candidates.pop(idx)
                    seg = cd['path'][::-1] if rev else cd['path']
                    chain = chain + seg if app_end else seg + chain
                    changed = True
            return chain

        large = [c for c in components if c['is_large']]
        small = [c for c in components if not c['is_large']]

        # Start chain from the largest component
        chain = list(large[0]['path']) if large else list(components[0]['path'])

        if len(large) > 1:
            chain = greedy_attach(chain, large[1:], self.max_gap_voxels)

        if self.chain_small_fragments and small:
            chain = greedy_attach(chain, small, self.max_small_gap_voxels)

        return np.array(chain)

    def _extract_points(self, skeleton_arr, img):
        ordered_indices = self._order_skeleton_points(skeleton_arr)
        center_points_physical = [
            img.TransformIndexToPhysicalPoint(tuple(map(int, pt)))
            for pt in ordered_indices
        ]
        return center_points_physical

    # ---- Step 3: smoothing spline fit ----
    def _fit_curve(self, pts, mesh_polyline=False):
        # Arc-length parametrization
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(d)])
        t /= t[-1]

        # Remove duplicates in t that would break splprep
        _, unique_idx = np.unique(t, return_index=True)
        pts = pts[unique_idx]
        t = t[unique_idx]

        k = min(3, len(pts) - 1)
        # s = smoothing_factor * N: total squared residual budget across all points.
        # At s=1.0 each point is allowed ~1 unit of residual on average.
        s = self.spline_smoothing * len(pts)
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], u=t, s=s, k=k)

        # mesh_polyline=True: slight extrapolation so the clipping step can
        # recover endpoints that skeletonize may have eroded.
        u_new = np.linspace(-0.05, 1.05, self.dense_samples) if mesh_polyline \
            else np.linspace(0.0, 1.0, self.dense_samples)

        coords = splev(u_new, tck, ext=0)
        return np.column_stack(coords)

    # ---- Step 4: clip with mesh ----
    def _clip_with_mesh(self, curve_pts, mesh):
        polyline = _vtk_polyline_from_points(curve_pts)
        clipped = _vtk_clip_polyline_with_mesh(polyline, mesh)
        return clipped

    # ---- Step 5: Full run ----
    def run(self, array_mask, nii_path):

        array_mask = np.swapaxes(array_mask, 0, -1)

        img, skeleton_arr, skeleton_sitk = self._load_and_skeleton(array_mask, nii_path)
        center_points_physical = self._extract_points(skeleton_arr, img)
        pts = np.asarray(center_points_physical)

        # Not enough points to fit a spline — return empty polydata so callers
        # can detect the missing curve via GetNumberOfPoints() == 0.
        if len(pts) < 2:
            return vtk.vtkPolyData()

        # Always extrapolate slightly beyond the skeleton endpoints so that
        # skeletonize-induced tip erosion is recovered; mesh clipping below
        # trims the excess back to the actual mask boundary.
        curve = self._fit_curve(pts, mesh_polyline=True)

        mesh = vtk_meshing(img)
        clipped = self._clip_with_mesh(curve, mesh)
        segments = _vtk_polyline_to_linesegments(clipped)

        return segments