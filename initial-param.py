import time
from pathlib import Path

import numpy as np
import vtk.util.numpy_support

OUT = Path("./out")
OUT.mkdir(exist_ok=True)

mesh_path = OUT.joinpath("mesh.vtk")

reader = vtk.vtkPolyDataReader()
reader.SetFileName(str(mesh_path))
reader.Update()
data: vtk.vtkPolyData = reader.GetOutput()
pdata: vtk.vtkPointData = data.GetPointData()

latitude = pdata.GetAbstractArray("Latitude")
longitude = pdata.GetAbstractArray("Longitude")
# normals = pdata.GetNormals()

NPOINTS = data.GetNumberOfPoints()
NCELLS = data.GetNumberOfCells()

lat = vtk.util.numpy_support.vtk_to_numpy(latitude)
lon = vtk.util.numpy_support.vtk_to_numpy(longitude)

cdata: vtk.vtkCellData = data.GetCellData()

cells = np.zeros((NCELLS, 4), dtype="i")
for idx in range(data.GetNumberOfCells()):
    cell: vtk.vtkCell = data.GetCell(idx)

    ids: vtk.vtkIdList = cell.GetPointIds()
    cells[idx] = [ids.GetId(k) for k in range(ids.GetNumberOfIds())]

sphere = np.array(
    [
        np.sin(lat) * np.cos(lon),
        np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ]
).T

# Compute areas for the whole dataset.
# Currently takes on the order of 25ms for 17000 vertices. Could probably be improved a bit by avoiding
# large temporary numpy objects, especially in the diag_a, diag_b, and cosines calculations.

print(sphere.shape)

start = time.time()

angle_det_indices = [
    [3, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 0],
]

diag_a_indices = [1, 0, 1, 0]
diag_b_indices = [3, 2, 3, 2]

corners = sphere[cells, :]

diag_a = corners[:, diag_a_indices]
diag_b = corners[:, diag_b_indices]
dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (diag_b * corners).sum(-1)

spats = np.linalg.det(corners[:, angle_det_indices])

areas = np.arctan2(dots, spats).sum(-1)
areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi

end = time.time()

print(end - start)
