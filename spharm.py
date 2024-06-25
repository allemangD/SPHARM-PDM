"""
Compute the initial parametrization from laplacian matrices of edge graph.
"""

import subprocess
from pathlib import Path

import numpy as np
import vtk
import vtk.util.numpy_support
from scipy.sparse import csgraph
from scipy.sparse import dok_array
from scipy.sparse import find
from scipy.sparse.linalg import spsolve

# MESH_ROOT = Path('~/Documents/chop/output/GenParaMesh/').expanduser()
# LABEL_ROOT = Path(
#     '~/Documents/chop/data/HLHS/Comprehensive Fontan Validated Segmentations/labels_post_processed').expanduser()
#
# mesh_path = MESH_ROOT.joinpath('8844-0012-01-STAGE3_tricuspid_septal_leaflet_surf.vtk').resolve()
# label_path = LABEL_ROOT.joinpath('8844-0012-01_tricuspid_septal_leaflet.nii.gz').resolve()

label_path = Path(
    "~/Documents/chop/data/HLHS/Comprehensive Fontan Validated Segmentations/labels_post_processed/8844-0012-01_tricuspid_septal_leaflet.nii.gz"
)
# label_path = Path('~/src/SPHARM-PDM/sample/duck.nii.gz')
# label_path = Path('~/src/SPHARM-PDM/sample/two-voxel.nii.gz')

label_path = label_path.expanduser().resolve()

OUT = Path("./out")
OUT.mkdir(exist_ok=True)

reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(str(label_path))

reader.Update()
raw = reader.GetOutput()

raw: vtk.vtkImageData
ext = list(raw.GetExtent())
ext[0] -= 1
ext[1] += 1
ext[2] -= 1
ext[3] += 1
ext[4] -= 1
ext[5] += 1

pad = vtk.vtkImageConstantPad()
pad.SetConstant(0)
pad.SetOutputWholeExtent(ext)
pad.SetInputData(raw)

net = vtk.vtkSurfaceNets3D()
net.SetInputConnection(pad.GetOutputPort())
net.SetSmoothing(False)
net.SetOutputStyleToBoundary()

clean = vtk.vtkCleanPolyData()
clean.SetInputConnection(net.GetOutputPort())
clean.PointMergingOn()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(clean.GetOutputPort())
normals.ComputeCellNormalsOff()
normals.ComputePointNormalsOn()
normals.SplittingOff()
normals.AutoOrientNormalsOn()
normals.FlipNormalsOff()
normals.ConsistencyOn()

feat = vtk.vtkExtractEdges()
feat.SetInputConnection(normals.GetOutputPort())
feat.UseAllPointsOn()
feat.Update()

ed: vtk.vtkPolyData = feat.GetOutput()
ed.BuildLinks()


def neighbors(data: vtk.vtkPolyData, pt: int) -> set:
    """Returns the set of point ids that share an edge with the given point id."""
    cell_ids = vtk.vtkIdList()
    data.GetPointCells(pt, cell_ids)
    point_ids = set()
    for cell_id_idx in range(cell_ids.GetNumberOfIds()):
        cell: vtk.vtkCell = data.GetCell(cell_ids.GetId(cell_id_idx))
        cell_point_ids: vtk.vtkIdList = cell.GetPointIds()
        for cell_point_id_idx in range(cell_point_ids.GetNumberOfIds()):
            point_ids.add(cell_point_ids.GetId(cell_point_id_idx))
    point_ids.remove(pt)
    return point_ids


def extract_points(data: vtk.vtkPolyData) -> np.ndarray:
    """Returns point coordinates in each row of a matrix."""
    res = np.zeros((data.GetNumberOfPoints(), 3))
    for idx in range(data.GetNumberOfPoints()):
        data.GetPoint(idx, res[idx])
    return res


def extract_normals(data: vtk.vtkPolyData) -> np.ndarray:
    res = np.zeros((data.GetNumberOfPoints(), 3))
    for idx in range(data.GetNumberOfPoints()):
        pts: vtk.vtkPointData = data.GetPointData()
        normals: vtk.vtkFloatArray = pts.GetNormals()
        res[idx] = normals.GetTuple(idx)
    return res


count = ed.GetNumberOfPoints()

# These aren't exactly constants, they are named indices, but treat them like constants anyway.
north = 0
south = count - 1

adjacency_matrix = dok_array((count, count))
for pt in range(count):
    for n in neighbors(ed, pt):
        adjacency_matrix[pt, n] = 1
adjacency_matrix = adjacency_matrix.tocsr()

# noinspection PyTypeChecker
# csgraph.laplacian is incorrectly type-hinted.
laplacian_matrix = csgraph.laplacian(
    # Sparse array support REQUIRES scipy>=1.11.3.
    adjacency_matrix
).tocsr()

# region Latitude problem.
lat = np.zeros((count,))
lat[south] = np.pi

# `[1:-1]` mask to avoid the poles; those are the boundary conditions.
# `values` is `pi` for nodes adjacent to the south pole and 0 elsewhere. Multiply trick keeps things sparse.
values = adjacency_matrix[:, [south]] * lat[south]
lat[1:-1] = spsolve(laplacian_matrix[1:-1, 1:-1], values[1:-1])
# endregion

# region Longitude problem.
lon = np.zeros((count,))

geo = vtk.vtkDijkstraGraphGeodesicPath()
geo.SetInputData(ed)
geo.SetStartVertex(south)
geo.SetEndVertex(north)
geo.Update()
short_path: np.ndarray = np.array(
    [geo.GetIdList().GetId(idx) for idx in range(geo.GetIdList().GetNumberOfIds())]
)

verts = extract_points(ed)
norms = extract_normals(ed)

values = np.zeros((count,))
for prv, idx, nxt in np.lib.stride_tricks.sliding_window_view(short_path, 3):
    I, _, _ = find(adjacency_matrix[:, [idx]])
    for n in I:
        if n in short_path:
            # don't alter the path itself
            continue

        # if west
        if np.dot(norms[idx], np.cross(verts[nxt] - verts[idx], verts[n] - verts[idx])) > 0:
            values[n] += 2 * np.pi
            values[idx] -= 2 * np.pi

lon_laplacian_matrix = laplacian_matrix.copy()
I, _, _ = find(adjacency_matrix[:, [north, south]])
for n in I:
    lon_laplacian_matrix[n, n] -= 1
lon_laplacian_matrix[0, 0] += 2

lon[1:-1] = spsolve(lon_laplacian_matrix[1:-1, 1:-1], values[1:-1])
# endregion

out = vtk.vtkPolyData()
out.DeepCopy(normals.GetOutput())

pd: vtk.vtkPointData = out.GetPointData()

arr = vtk.util.numpy_support.numpy_to_vtk(lat)
arr.SetName("Latitude")
pd.AddArray(arr)

arr = vtk.util.numpy_support.numpy_to_vtk(lon)
arr.SetName("Longitude")
pd.AddArray(arr)

writer = vtk.vtkPolyDataWriter()
writer.SetFileName(str(OUT / "mesh.vtk"))
writer.SetInputData(out)
writer.Update()

subprocess.run(
    [
        "f3d",
        OUT / "mesh.vtk",
        "--output",
        OUT / "lat.png",
        "--resolution",
        "1200,1000",
        "--edges",
        "--scalars=Latitude",
    ]
)

subprocess.run(
    [
        "f3d",
        OUT / "mesh.vtk",
        "--output",
        OUT / "lon.png",
        "--resolution",
        "1200,1000",
        "--edges",
        "--scalars=Longitude",
    ]
)
