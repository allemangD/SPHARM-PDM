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

############################################################

areas_each = []
spats_each = []

for cell in cells:
    corners = sphere[cell, :]

    diag_a = corners[[1, 0, 1, 0]]
    diag_b = corners[[3, 2, 3, 2]]
    diag_dot = diag_a * diag_b
    a_dot = diag_a * corners
    b_dot = diag_b * corners
    cosines = diag_dot.sum(-1) - a_dot.sum(-1) * b_dot.sum(-1)

    # 1 . 3 - 1 . 0 * 0 . 3
    # 0 . 2 - 0 . 1 * 1 . 2
    # 1 . 3 - 1 . 2 * 2 . 3
    # 0 . 2 - 0 . 3 * 3 . 2

    # spat is the old inequality list
    spat = np.linalg.det(
        corners[
            [
                [3, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 0],
            ]
        ]
    )

    area = np.arctan2(cosines, spat).sum()
    area = np.fmod(area + 8.5 * np.pi, np.pi) - 0.5 * np.pi

    areas_each.append(area)
    spats_each.append(spat)

areas_each = np.array(areas_each)

print(cells.shape)
print(sphere.shape)
corners = sphere[cells, :]

diag_a = corners[:, [1, 0, 1, 0]]
diag_b = corners[:, [3, 2, 3, 2]]
cosines = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (
    diag_b * corners
).sum(-1)

spats = np.linalg.det(
    corners[
        :,
        [
            [3, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 0],
        ],
    ]
)

areas = np.arctan2(cosines, spats).sum(-1)
areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi

print(np.allclose(areas, areas_each))
print(np.allclose(spats, spats_each))

# corners[:, [3, 0, 1]]
# print(corners.shape)
# print(corners)

# now compute everything in one go with np...
