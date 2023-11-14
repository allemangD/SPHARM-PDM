from pathlib import Path
import subprocess

import vtk
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
from scipy.sparse import csgraph
from scipy.sparse import dok_array
from scipy.sparse import csr_array
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

feat = vtk.vtkExtractEdges()
feat.SetInputConnection(clean.GetOutputPort())
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

# Solve for latitude problem. sub_matrix contains elements for all nodes _except_ the poles.
lat = np.zeros((count,))
lat[south] = np.pi

values: csr_array = adjacency_matrix[:, [south]] * lat[south]
result = spsolve(laplacian_matrix[1:-1, 1:-1], values[1:-1])
lat[1:-1] = result

# build the longitude matrix

# for n in neighbors(ed, NORTH):
#     if NORTH < n < SOUTH:
#         arr[n - 1, n - 1] -= 1
#
# lon_arr = scipy.sparse.csr_array(arr)
# lon = scipy.sparse.linalg.spsolve(lon_arr, b)

# c = np.zeros((count - 2,))
# prev = north
# curr = next(iter(lut[north]))
# max_ = 0.0
# while curr < south:
#     for n in lut[curr]:
#         if lat[n - 1] > max_:
#             next_ = n
#         if n == prev:
#             prev_ = n
#
# lon_arr = scipy.sparse.csr_array(arr)
# lon = scipy.sparse.linalg.spsolve(lon_arr, c)

out = vtk.vtkPolyData()
out.DeepCopy(clean.GetOutput())

pd: vtk.vtkPointData = out.GetPointData()

arr = vtk.vtkFloatArray()
arr.SetName("Latitude")
arr.InsertNextValue(0)
for i, v in enumerate(lat):
    arr.InsertNextValue(v)
arr.InsertNextValue(np.pi)
pd.AddArray(arr)

geo = vtk.vtkDijkstraGraphGeodesicPath()
geo.SetInputData(ed)
geo.SetStartVertex(north)
geo.SetEndVertex(south)
geo.Update()
short_path: vtk.vtkIdList = geo.GetIdList()

arr = vtk.vtkFloatArray()
arr.SetName("Longitude")
arr.SetNumberOfValues(ed.GetNumberOfPoints())
arr.Fill(0)
for idx in range(short_path.GetNumberOfIds()):
    arr.SetValue(short_path.GetId(idx), 1)
# arr.InsertNextValue(0)
# for i, v in enumerate(lon):
#     arr.InsertNextValue(i)
# arr.InsertNextValue(0)
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
