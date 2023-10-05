import subprocess

import vtk

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from pathlib import Path

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


def neighbors(data: vtk.vtkPolyData, pt: int):
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


# build latitude system
lat_arr = scipy.sparse.dok_array(
    (
        ed.GetNumberOfPoints() - 2,
        ed.GetNumberOfPoints() - 2,
    )
)

for pt in range(1, ed.GetNumberOfPoints() - 1):
    ns = neighbors(ed, pt)
    for n in ns:
        if 0 < n < ed.GetNumberOfPoints() - 1:
            lat_arr[pt - 1, n - 1] = -1
    lat_arr[pt - 1, pt - 1] = len(ns)

b = np.zeros((ed.GetNumberOfPoints() - 2,))
for n in neighbors(ed, ed.GetNumberOfPoints() - 1):
    if 0 < n < ed.GetNumberOfPoints() - 1:
        b[n - 1] = np.pi

# solve the systems
lat_arr = scipy.sparse.csr_array(lat_arr)

lat = scipy.sparse.linalg.spsolve(lat_arr, b)

out = vtk.vtkPolyData()
out.DeepCopy(clean.GetOutput())

pd: vtk.vtkPointData = out.GetPointData()
arr = vtk.vtkFloatArray()
arr.SetName("Latitude")
arr.InsertNextValue(0)
for v in enumerate(lat):
    arr.InsertNextValue(v)
arr.InsertNextValue(np.pi)
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
        OUT / "mesh.png",
        "--resolution",
        "1024,1280",
        "-e",
    ]
)
