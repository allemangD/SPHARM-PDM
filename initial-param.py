"""
Apply equal-area and angle optimizations to the parametrization.

This processes the output of spharm.py.
"""

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

ANGLE_DET_INDICES = [
    [3, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 0],
]

DIAG_A_INDICES = [1, 0, 1, 0]
DIAG_B_INDICES = [3, 2, 3, 2]

IDEAL_CELL_AREA = 4 * np.pi / cells.shape[0]


def area_variance(x):
    print('comp')
    cart = x.reshape(sphere.shape)
    corners = cart[cells, :]

    diag_a = corners[:, DIAG_A_INDICES]
    diag_b = corners[:, DIAG_B_INDICES]
    dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (diag_b * corners).sum(-1)

    spats = np.linalg.det(corners[:, ANGLE_DET_INDICES])

    areas = np.arctan2(dots, spats).sum(-1)
    areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi

    variance = ((areas - IDEAL_CELL_AREA) * (areas - IDEAL_CELL_AREA)).sum()

    return variance

def area_variance_jac(x):
    cart = x.reshape(sphere.shape)
    corners = cart[cells, :]


import scipy.optimize

print('about to optimize')
res = scipy.optimize.minimize(
    area_variance,
    sphere.ravel(),
    method="newton-cg",
    tol=30,
    options={"maxiter": 1, 'verbose': True},
    jac='2-point',
    hess=scipy.optimize.BFGS(),

)
print(res)

# The constraints:

# minimize:
# constrain

# variance' = 2 * (areas -

# `spats > 0`  # (?) not sure if this is > or <
# `areas == UNIT_SPHERE_SURFACE_AREA / sphere.shape[0]`


# See https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate
# -scalar-functions-minimize section on defining constraints. We are optimizing the [sphere] positions s.t.
# `spats>0` and `areas=4pi/n` as defined above. So we need nonlinear constraints, so we need a jacobian and
# a hessian. See if there's some way to autodiff those or if I need to manually do that part. The
# computations aren't _so_ terrible so I might be able to manually compute it.

# What is the function to optimize? We have some constraints: spats>0 and areas=4pi/n. But it's unclear
# what the actual function to be minimized is.

# we want to minimize
#  - the variance of areas: (x - Ex) ^ 2.
#  - this is implicitly positive.
# we want to constrain
#  - 0 <= areas must be positive.
