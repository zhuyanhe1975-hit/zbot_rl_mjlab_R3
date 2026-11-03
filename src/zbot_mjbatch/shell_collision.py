"""Opt-in convex-hull clipping for connector collision experiments.

This only creates a collision shell and leaves the original mesh for display.
It does not add a mating disk or holes, update explicit contact pairs, excludes,
collision audits, or make any claim of complete-robot collision coverage.
Removing the end slab changes the collision envelope; callers must validate it.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class ShellCut:
    """A geom and a site on its body; retain the site's +Z or -Z halfspace.

    ``direction`` is +1 or -1 and ``depth_m`` is measured from the site plane
    into that side of the module, for example 0.002 for a 2 mm end disk.
    """

    geom: str
    site: str
    direction: int
    depth_m: float = .002


def _rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _find_exact(elements, name):
    # MjSpec's name index can be stale after inserting elements on a compiled
    # spec. Iteration reads the actual current names instead of cached IDs.
    return next((element for element in elements if element.name == name), None)


def _clipped_hull(vertices, origin, normal, depth):
    """Intersect an existing convex hull with a plane; return oriented mesh."""
    vertices = np.asarray(vertices, dtype=float)
    original = ConvexHull(vertices)
    triangles = original.simplices
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]],
                                             triangles[:, [1, 2]],
                                             triangles[:, [2, 0]])), axis=1), axis=0)
    signed = (vertices - origin) @ normal - depth
    keep = vertices[signed >= 0.]
    a, b = edges[:, 0], edges[:, 1]
    crossing = ((signed[a] < 0.) & (signed[b] > 0.)) | ((signed[a] > 0.) & (signed[b] < 0.))
    a, b = a[crossing], b[crossing]
    fraction = signed[a] / (signed[a] - signed[b])
    intersections = vertices[a] + fraction[:, None] * (vertices[b] - vertices[a])
    points = np.vstack((keep, intersections))
    if len(points) < 4 or not np.any(signed < 0.) or not np.any(signed > 0.):
        raise ValueError('cut plane must cross the original collision hull')
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    normal_from_faces = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                 points[faces[:, 2]] - points[faces[:, 0]])
    flipped = np.einsum('ij,ij->i', normal_from_faces, hull.equations[:, :3]) < 0.
    faces[flipped] = faces[flipped][:, [0, 2, 1]]
    index = np.full(len(points), -1, dtype=int)
    index[hull.vertices] = np.arange(len(hull.vertices))
    return points[hull.vertices], index[faces]


def add_clipped_collision_shells(spec, cuts):
    """Turn existing mesh geoms into visuals and add plane-clipped shells.

    The site must be rigidly attached to the original geom's body. The current
    compiled MuJoCo mesh vertices, including compiler mesh transforms, are used,
    so the result is the old convex collision hull clipped in body coordinates.
    New geoms have zero mass; existing explicit body inertials are preserved.

    IMPORTANT: MuJoCo explicit contact pairs can bypass a visual geom's zero
    contype/conaffinity. Existing project Python/C++ audits may also enumerate
    visuals directly. Callers must separately rebuild pair/exclude rules and
    audits before use in a simulation; this helper does not touch those rules.

    Returns a list of dictionaries identifying the visual and collision geoms.
    """
    cuts = tuple(cuts)
    if not cuts:
        return []
    if len({c.geom for c in cuts}) != len(cuts):
        raise ValueError('each original geom may appear only once')
    model = spec.compile()
    prepared = []
    for cut in cuts:
        if cut.direction not in (-1, 1) or not np.isfinite(cut.depth_m) or cut.depth_m <= 0.:
            raise ValueError('direction must be +1/-1 and depth must be positive and finite')
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cut.geom)
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cut.site)
        if g < 0 or s < 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f'unknown mesh geom or site: {cut.geom}, {cut.site}')
        if model.geom_bodyid[g] != model.site_bodyid[s]:
            raise ValueError('geom and site must belong to the same body')
        name = cut.geom + '_clipped_shell'
        meshname = cut.geom + '_clipped_shell_mesh'
        if _find_exact(spec.geoms, name) is not None or _find_exact(spec.meshes, meshname) is not None:
            raise ValueError(f'clipped shell name already exists: {name}')
        mesh = model.geom_dataid[g]
        address, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        # Compiled mesh vertices are centered and rotated by the mesh compiler;
        # geom_pos/quat map them back to their original body's coordinates.
        vertices = model.mesh_vert[address:address + count] @ _rotation(model.geom_quat[g]).T + model.geom_pos[g]
        normal = cut.direction * _rotation(model.site_quat[s])[:, 2]
        v, f = _clipped_hull(vertices, model.site_pos[s], normal, cut.depth_m)
        prepared.append((cut, g, name, meshname, v, f))
    result = []
    for cut, g, name, meshname, vertices, faces in prepared:
        parent = _find_exact(spec.geoms, cut.geom).parent
        spec.add_mesh(name=meshname, uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
        parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshname,
                        mass=0., contype=int(model.geom_contype[g]),
                        conaffinity=int(model.geom_conaffinity[g]), condim=int(model.geom_condim[g]),
                        priority=int(model.geom_priority[g]), friction=model.geom_friction[g].tolist(),
                        solref=model.geom_solref[g].tolist(), solimp=model.geom_solimp[g].tolist(),
                        margin=float(model.geom_margin[g]), gap=float(model.geom_gap[g]),
                        rgba=(0., 0., 0., 0.))
        # Adding geoms may invalidate previous MjSpec element views.
        old = _find_exact(spec.geoms, cut.geom)
        old.contype = 0
        old.conaffinity = 0
        result.append({'visual_geom': cut.geom, 'collision_geom': name,
                       'site': cut.site, 'direction': cut.direction, 'depth_m': cut.depth_m})
    return result
