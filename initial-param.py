"""
Apply equal-area and angle optimizations to the parametrization.

This processes the output of spharm.py.
"""

from pathlib import Path

import scipy.sparse
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import NonlinearConstraint

import numpy as np
import vtk.util.numpy_support
from scipy.sparse import csr_array, dok_array


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


# mesh_path = Path('sample/duck.vtk')
# mesh_path = Path('sample/two-voxel.vtk')

mesh_path = Path('sample/hourglass_seg.vtk')
# mesh_path = Path('sample/cilinder_seg.vtk')

# mesh_path = Path('sample/hippocampus.vtk')

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

# print(sphere.shape)

IDEAL_CELL_AREA = 4 * np.pi / cells.shape[0]

EDGES = [[0, 1], [1, 2], [2, 3], [3, 0]]

ANGLE_DET_INDICES = [
    [3, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 0],
]

DIAG_A_INDICES = [1, 0, 1, 0]
DIAG_B_INDICES = [3, 2, 3, 2]


# todo: let `x` be the spherical coordinates; then there is no need for norm constraint. would need
#   to recompute `sphere` in goal_func, so better to merge `gradient` into that with `jac=True`.
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

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


def norms(x) -> np.ndarray:
    """To constrain norm of each vectors to 1."""

    points = x.reshape(sphere.shape)

    # norm = np.linalg.norm(points, axis=1)
    # return np.tile(norm, (1, 3)).ravel()

    norm = np.linalg.norm(points, axis=1)
    return norm


def norms_jac(x) -> np.ndarray:
    """To constrain norm of each vector to 1."""

    # Jacobian (gradient) at any point is normal to the sphere at that point.

    points = x.reshape(sphere.shape)
    norm = np.linalg.norm(points, axis=1)
    normalized = points / norm[:, None]

    idxs = np.arange(len(points))

    res = dok_array((len(points), len(x)))
    for k in range(points.shape[1]):
        res[idxs, points.shape[1] * idxs + k] = normalized[:, k]

    return res
    # res = csr_array((len(points), len(x)))
    # res[idxs, idxs * 3] = 1
    # # res[]
    #
    # # should return a (V, 3V) array.
    # # each col is a constraint;
    #
    # # return normalized.ravel()
    # return normalized.ravel().reshape(1, -1)

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

    print(areas.shape)
    # return areas
    return areas - IDEAL_CELL_AREA  # constrain eq 0


# constraints are only supported with:
# COBYLA, COBYQA, SLSQP, trust-constr
#
# object constraints for cobyqa, trust-constr
#
# dict constraints for cobyla, slsqp

# - cobyla, cobyqa do not use gradient
#   - constraints of type 'eq' not handled by la

# - slsqp fast but out of memory on big mesh
# - trust-constr is all that's left. it works fine for the small mesh `duck` but ran for 10 hours
#     without terminating on a tricuspid leaflet.
#     todo play with tolerances to get it to terminate faster? MAYBE this will work?

print('about to minimize')

# print(norms_jac(sphere.ravel()).shape)


def cb(xi, res: OptimizeResult):
    # print(res)
    # print(res.nit, res.nfev, res.njev, res.constr_nfev, res.constr_njev, res.constr_nhev)
    print(res)
    # print('constr min', *[c.min() for c in res.constr], 'max', *[c.max() for c in res.constr])
    print()

res = minimize(
    goal_func,
    sphere.ravel(),

    # callback=cb,

    jac=gradient,
    hess='2-point',
    constraints=[
        NonlinearConstraint(
            norms, 1, 1,
            jac=norms_jac,
            hess='2-point',
        ),
    ],

    # method="COBYQA",
    # options=dict(
    #     maxiter=5,
    #     # maxfev=1000,
    #     disp=True,
    # ),

    # method='SLSQP',
    # options=dict(
    #     maxiter=50,
    #     disp=True,
    # ),

    method='trust-constr',
    options=dict(
        maxiter=10,
        # xtol=1,
        # gtol=1,
        verbose=2,
        # sparse_jacobian=True,
        # xtol=1e-2,
        # verbose=2,
    ),

)
print(res)
result = res.x.reshape(sphere.shape)
# print(result)
# print(areas(res.x))

# print(areas(result.ravel()))
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
