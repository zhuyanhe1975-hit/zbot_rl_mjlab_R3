"""Standalone pin/socket collision-geometry prototype; not a robot model.

All pin dimensions and placement are caller-provided, not measurements. The
socket plate is a polygonal approximation decomposed into convex polygonal
prisms; adding a solid disc underneath would obstruct the holes.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


@dataclass(frozen=True)
class ConnectorGeometry:
    pin_radius_m: float
    bolt_radius_m: float
    radial_clearance_m: float
    disk_radius_m: float = .05
    disk_thickness_m: float = .002
    pin_height_m: float = .002
    outer_segments: int = 64
    hole_segments: int = 16

    def __post_init__(self):
        values = (self.pin_radius_m, self.bolt_radius_m, self.radial_clearance_m,
                  self.disk_radius_m, self.disk_thickness_m, self.pin_height_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('all geometry dimensions must be finite and positive')
        if self.outer_segments < 16 or self.hole_segments < 8:
            raise ValueError('insufficient polygon resolution')
        hole_vertex_radius = self.hole_radius_m / math.cos(math.pi / self.hole_segments)
        # Bound by the disk polygon's apothem, not only by its vertices.
        if self.bolt_radius_m + hole_vertex_radius >= self.disk_radius_m * math.cos(math.pi / self.outer_segments):
            raise ValueError('holes must lie strictly inside the disc polygon')
        if math.sqrt(2.) * self.bolt_radius_m <= 2. * hole_vertex_radius:
            raise ValueError('hole polygons must not overlap')

    @property
    def hole_radius_m(self):
        return self.pin_radius_m + self.radial_clearance_m

    @property
    def centers(self):
        a = np.arange(4) * (math.pi / 2.)
        return self.bolt_radius_m * np.column_stack((np.cos(a), np.sin(a)))


def socket_triangles(config):
    """Return a conforming polygonal triangulation of the plate with 4 holes.

    Hole polygon inradius equals the requested circular hole radius, so it
    doesn't inadvertently make the aperture smaller than specified. Fail if
    Delaunay fails to preserve any boundary or if area is not conserved.
    """
    outer_angles = np.arange(config.outer_segments) * 2 * math.pi / config.outer_segments
    points = [config.disk_radius_m * np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))]
    hole_angles = (np.arange(config.hole_segments) + .5) * 2 * math.pi / config.hole_segments
    vertex_radius = config.hole_radius_m / math.cos(math.pi / config.hole_segments)
    for center in config.centers:
        points.append(center + vertex_radius * np.column_stack((np.cos(hole_angles), np.sin(hole_angles))))
    coords = np.concatenate(points)
    triangulation = Delaunay(coords)
    simplices = triangulation.simplices
    edges = {tuple(sorted((int(a), int(b)))) for t in simplices
             for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    start = 0
    for boundary in points:
        n = len(boundary)
        if any(tuple(sorted((start+i, start+(i+1)%n))) not in edges for i in range(n)):
            raise ValueError('triangulation does not preserve a socket boundary; adjust dimensions or polygon resolution')
        start += n
    def in_convex(poly, point):
        direction = np.roll(poly, -1, axis=0) - poly
        diff = point - poly
        cross = direction[:, 0] * diff[:, 1] - direction[:, 1] * diff[:, 0]
        return np.all(cross >= -1e-14)
    retained = [coords[t] for t in simplices
                if not any(in_convex(h, coords[t].mean(axis=0)) for h in points[1:])]
    def area(poly):
        return abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                - poly[:, 1] * np.roll(poly[:, 0], -1)))) / 2.
    expected = area(points[0]) - sum(area(h) for h in points[1:])
    actual = sum(area(t) for t in retained)
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-14):
        raise ValueError('socket triangulation has missing or overlapping area')
    return np.asarray(retained)


def merge_convex_triangles(triangles):
    """Join adjacent triangles only when their exact union remains convex."""
    polys = [np.asarray(t) for t in triangles]
    def area(poly):
        return abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                - poly[:, 1] * np.roll(poly[:, 0], -1)))) / 2.
    while True:
        adjacency = {}
        for i, polygon in enumerate(polys):
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                key = tuple(sorted((tuple(a), tuple(b))))
                adjacency.setdefault(key, []).append(i)
        joined = False
        for indexes in adjacency.values():
            if len(indexes) != 2:
                continue
            i, j = indexes
            vertices = np.unique(np.vstack((polys[i], polys[j])), axis=0)
            hull = ConvexHull(vertices)
            if np.isclose(hull.volume, area(polys[i]) + area(polys[j]), rtol=1e-10, atol=1e-16):
                polys[i] = vertices[hull.vertices]
                del polys[j]
                joined = True
                break
        if not joined:
            return polys


def inject_connector_collision(spec, pin_site_name, socket_site_name, config, *,
                               pin_center_site, socket_center_site, prefix):
    """Add pin and socket collision geometry in the two site-local frames.

    Site-relative disk-center offsets are mandatory: existing port sites are
    not necessarily at the CAD mounting center. Both disks use local +Z as
    their shared registration axis: the pin plate occupies +Z and its pins
    extend in -Z; the socket occupies -Z. The caller must supply site frames
    already representing the same registration orientation at mating.

    Geoms and meshes are attached to each site's current parent body pose;
    they do not follow subsequent runtime edits to the site model fields. This
    does not alter existing geoms, equality constraints, filters or exclusions.
    Added geoms have zero mass to preserve pre-existing body inertias. The
    caller must resolve overlapping original collision hulls, contact filters,
    and site conventions before using this in an existing robot scene.
    """
    if not prefix or not isinstance(prefix, str):
        raise ValueError('prefix must be a nonempty string')
    # MjSpec cached name lookups can be stale after appending elements to a
    # previously compiled spec; iterate the current collection by exact name.
    pin_site = next((site for site in spec.sites if site.name == pin_site_name), None)
    socket_site = next((site for site in spec.sites if site.name == socket_site_name), None)
    if pin_site is None or socket_site is None:
        raise ValueError('both connector sites must exist in the spec')
    if pin_site.parent.name == socket_site.parent.name:
        raise ValueError('connector sites must belong to distinct bodies')
    def frame(site, offset):
        offset = np.asarray(offset, dtype=float)
        quat = np.asarray(site.quat, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('site-local center must have 3 finite components')
        if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1., atol=1e-6):
            raise ValueError('site quaternion must be finite and normalized')
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        return np.asarray(site.pos, dtype=float) + rotation @ offset, rotation, quat.copy()
    pin_origin, pin_rot, pin_quat = frame(pin_site, pin_center_site)
    socket_origin, socket_rot, socket_quat = frame(socket_site, socket_center_site)
    pin_body, socket_body = pin_site.parent, socket_site.parent
    height = config.disk_thickness_m
    def point(origin, rotation, vector):
        return origin + rotation @ np.asarray(vector, dtype=float)
    pin_body.add_geom(name=f'{prefix}_pin_plate_disc', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=point(pin_origin, pin_rot, (0, 0, height/2)), quat=pin_quat,
        size=(config.disk_radius_m, height/2, 0), mass=0., rgba=(.10, .45, .80, .35))
    for i, (x, y) in enumerate(config.centers):
        pin_body.add_geom(name=f'{prefix}_pin_{i}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=point(pin_origin, pin_rot, (x, y, -config.pin_height_m/2)), quat=pin_quat,
            size=(config.pin_radius_m, config.pin_height_m/2, 0), mass=0., rgba=(1., .68, .10, 1.))
    pieces = merge_convex_triangles(socket_triangles(config))
    for i, polygon in enumerate(pieces):
        n = len(polygon)
        verts = np.vstack((np.column_stack((polygon, np.full(n, -height))),
                           np.column_stack((polygon, np.zeros(n)))))
        faces = []
        for j in range(1, n-1):
            faces.extend(((0, j+1, j), (n, n+j, n+j+1)))
        for j in range(n):
            k = (j+1) % n
            faces.extend(((j, k, n+k), (j, n+k, n+j)))
        name = f'{prefix}_socket_piece_{i}'
        spec.add_mesh(name=name, uservert=verts.ravel().tolist(), userface=np.asarray(faces).ravel().tolist())
        socket_body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                             pos=socket_origin, quat=socket_quat, mass=0., rgba=(.72, .78, .84, 1.))
    return {'pin_body': pin_body.name, 'socket_body': socket_body.name,
            'pin_disc': f'{prefix}_pin_plate_disc',
            'pins': tuple(f'{prefix}_pin_{i}' for i in range(4)),
            'socket_pieces': tuple(f'{prefix}_socket_piece_{i}' for i in range(len(pieces)))}


def make_standalone_spec(config):
    """Build two free bodies with contacting discs; SI units; no weld or magnets.

    Female surface is z=0 at qpos0, male surface z=0. The male pins point
    in negative z and enter four through-holes. Geoms use explicit inertials,
    so changing tessellation does not change body mass. This is an isolated
    two-body mechanics prototype and is not added to the existing FSM.
    """
    spec = mujoco.MjSpec.from_string('''<mujoco model="connector_prototype">
      <option timestep="0.001" gravity="0 0 0"/>
      <statistic center="0 0 0.01" extent="0.15"/>
      <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/></visual>
      <worldbody>
        <light pos="0.1 -0.1 0.2" dir="-0.3 0.3 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="-0.1 0.1 -0.15" dir="0.3 -0.3 1" diffuse="0.6 0.6 0.6"/>
        <body name="socket"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/></body>
        <body name="pin_plate"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/></body>
      </worldbody></mujoco>''')
    socket = spec.body('socket'); pin_plate = spec.body('pin_plate')
    height = config.disk_thickness_m
    pin_plate.add_geom(name='pin_plate_disc', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=(0, 0, height / 2), size=(config.disk_radius_m, height / 2, 0),
                       rgba=(.10, .45, .80, .35))
    for i, (x, y) in enumerate(config.centers):
        pin_plate.add_geom(name=f'pin_{i}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           pos=(x, y, -config.pin_height_m / 2),
                           size=(config.pin_radius_m, config.pin_height_m / 2, 0),
                           rgba=(1., .68, .10, 1.))
    for i, polygon in enumerate(merge_convex_triangles(socket_triangles(config))):
        # ConvexHull returns the polygon in CCW order.
        n = len(polygon)
        verts = np.vstack((np.column_stack((polygon, np.full(n, -height))),
                           np.column_stack((polygon, np.zeros(n)))))
        faces = []
        for j in range(1, n-1):
            faces.extend(((0, j+1, j), (n, n+j, n+j+1)))
        for j in range(n):
            k = (j+1) % n
            faces.extend(((j, k, n+k), (j, n+k, n+j)))
        name = f'socket_piece_{i}'
        spec.add_mesh(name=name, uservert=verts.ravel().tolist(), userface=np.asarray(faces).ravel().tolist())
        socket.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                        rgba=(.72, .78, .84, 1.))
    return spec


def make_standalone_model(config):
    """Compile the isolated prototype built by :func:`make_standalone_spec`."""
    return make_standalone_spec(config).compile()
