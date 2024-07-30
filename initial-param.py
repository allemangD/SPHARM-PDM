"""
Apply equal-area and angle optimizations to the parametrization.

This processes the output of spharm.py.
"""

from pathlib import Path

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import numpy as np
import vtk.util.numpy_support


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

print(sphere.shape)

IDEAL_CELL_AREA = 4 * np.pi / cells.shape[0]

EDGES = [[0, 1], [1, 2], [2, 3], [3, 0]]


def goal_func(x) -> float:
    """
    See EqualAreaParametricMeshNewtonIterator::goal_func
    """

    # for vertex in vertices:
    #     for neighbor in neighbors(vertex):
    #         goal += 1 - dot(vertex, neighbor)

    points = x.reshape(sphere.shape)

    prod = np.prod(points[cells[:, EDGES]], axis=-2).reshape(-1, 3)
    goal = (len(prod) - prod.sum()) / 2
    return goal


def gradient(x) -> np.ndarray:
    """
    See EqualAreaParametricMeshNewtonIterator::calc_gradient
    """

    # for vertex in vertices:
    #     nbsum = [0, 0, 0]
    #     for neighbor in neighbors(vertex):
    #         nbsum += neighbor
    #     gradient[vertex] = dot(vertex, nbsum) * vertex - nbsum

    points = x.reshape(sphere.shape)

    nbsum = np.zeros_like(points)
    for u, v in EDGES:
        nbsum[cells[:, u]] += points[cells[:, v]]
        nbsum[cells[:, v]] += points[cells[:, u]]

    prod = (nbsum * points).sum(-1)

    return (points * prod[:, None] - nbsum).ravel()


ANGLE_DET_INDICES = [
    [3, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 0],
]

DIAG_A_INDICES = [1, 0, 1, 0]
DIAG_B_INDICES = [3, 2, 3, 2]


def norms(x) -> np.ndarray:
    """To constrain norm of each vectors to 1."""

    points = x.reshape(sphere.shape)
    # return np.linalg.norm(points, axis=1)
    return np.linalg.norm(points, axis=1) - 1  # constrain eq 0


def areas(x) -> np.ndarray:
    """To constrain area of each cell to 4pi/num_cells"""

    points = x.reshape(sphere.shape)
    corners = points[cells, :]

    diag_a = corners[:, DIAG_A_INDICES]
    diag_b = corners[:, DIAG_B_INDICES]
    dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (diag_b * corners).sum(-1)

    spats = np.linalg.det(corners[:, ANGLE_DET_INDICES])

    areas = np.arctan2(dots, spats).sum(-1)
    areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi

    # return areas
    return areas - IDEAL_CELL_AREA  # constrain eq 0


# print(areas(sphere.ravel()))
# print(norm(sphere.ravel()).shape)
# print(len(sphere))

# u, v = EDGES[0]
# print(cells[:, u])
# print(cells[:, v])
# exit()

# print(gradient(sphere.ravel()))
# exit()

# def area_variance(x):
#     cart = x.reshape(sphere.shape)
#     corners = cart[cells, :]
#
#     diag_a = corners[:, DIAG_A_INDICES]
#     diag_b = corners[:, DIAG_B_INDICES]
#     dots = (diag_a * diag_b).sum(-1) - (diag_a * corners).sum(-1) * (diag_b * corners).sum(-1)
#
#     spats = np.linalg.det(corners[:, ANGLE_DET_INDICES])
#
#     areas = np.arctan2(dots, spats).sum(-1)
#     areas = np.fmod(areas + 8.5 * np.pi, np.pi) - 0.5 * np.pi
#
#     variance = ((areas - IDEAL_CELL_AREA) * (areas - IDEAL_CELL_AREA)).sum()
#
#     return variance
#
# def area_variance_jac(x):
#     cart = x.reshape(sphere.shape)
#     corners = cart[cells, :]
#
#
# def jac(x):
#     cart = x.reshape(sphere.shape)
#
#     # return the gradient vector.
#     # d area_variance / d cart
#
#     # should be reshaped back to x.
#
#     # should probably be moved into `area_variance(x) -> float, vec` with `jac=True`.
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
#
#     # for each component in the input - how does the area variance change if we tweak that component?
#
#     # problem: right now x is the cartesian coordinates. This is wrong. I want to optimize
#     # the spherical parametrization. That should only be 2 components for each vertex, not 3.


res = minimize(
    goal_func,
    sphere.ravel(),
    jac=gradient,
    # constraints=[
    #     dict(type='eq', fun=norms),
    #     dict(type='eq', fun=areas),
    # ],
    # method='trust-constr',
    constraints=[
        NonlinearConstraint(norms, 0, 0),
        NonlinearConstraint(areas, 0, 0),
    ],
    method='trust-constr',
    options=dict(
        maxiter=50,
        # xtol=1e-1,
    ),
)
print(res)
result = res.x.reshape(sphere.shape)
# print(result - sphere)
print(result)
print(areas(result.ravel()))
# print(result)
# print(sphere)

# import scipy.optimize
#
# print('about to optimize')
# res = scipy.optimize.minimize(
#     area_variance,
#     sphere.ravel(),
#     constraints=(),
#     method="COBYLA",
#     options={"maxiter": 3000},
#     # method="newton-cg",
#     # tol=30,
#     # options={"maxiter": 1, 'verbose': True},
#     # jac='2-point',
#     # hess=scipy.optimize.BFGS(),
# )
# print(res)

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
