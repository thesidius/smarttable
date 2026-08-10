"""Forward model: a polyhedral die resting on the tray floor, projected.

No convex-hull library needed. These solids are duals of each other, so one's
FACE normals point along the other's VERTICES -- which is all that is required
to stand a die on a face and find its inradius.
"""
import numpy as np, cv2

PHI = (1 + 5 ** 0.5) / 2

def _icosa_v():
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [[0, s1, s2*PHI], [s1, s2*PHI, 0], [s2*PHI, 0, s1]]
    return np.unique(np.array(v, float), axis=0)

def _dodeca_v():
    v = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [[0, s1/PHI, s2*PHI], [s1/PHI, s2*PHI, 0], [s2*PHI, 0, s1/PHI]]
    return np.unique(np.array(v, float), axis=0)

def _cube_v():
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                     for z in (-1, 1)], float)

def _octa_v():
    return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)

def _octa_v():
    return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)


def _tetra_v():
    return np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)


def _trapez10_v(a=0.36):
    """Pentagonal trapezohedron -- the d10 and d%.

    Two rings of five, staggered by 36 degrees, plus two apexes. The apex height
    is not free: each kite face must be planar, which pins it given the ring
    height. Solved numerically rather than quoted, so changing `a` stays valid.

    `a` itself sets how squat the solid is and is chosen to match a real d10
    (roughly as tall as it is wide). Approximate -- unlike the Platonic solids
    there is no canonical proportion, and manufacturers differ.
    """
    ang_u = np.arange(5) * 2*np.pi/5
    ang_l = ang_u + np.pi/5
    up = np.stack([np.cos(ang_u), np.sin(ang_u), np.full(5, a)], 1)
    lo = np.stack([np.cos(ang_l), np.sin(ang_l), np.full(5, -a)], 1)

    def planar_err(c):
        A = np.array([0.0, 0.0, c])
        return float(np.linalg.det(np.stack([up[0]-A, lo[0]-A, up[1]-A])))
    c0, c1 = a + 1e-4, 12.0
    for _ in range(200):                       # bisect for a planar kite
        mid = (c0 + c1) / 2
        if planar_err(c0) * planar_err(mid) <= 0:
            c1 = mid
        else:
            c0 = mid
    c = (c0 + c1) / 2
    return np.vstack([up, lo, [[0, 0, c], [0, 0, -c]]])


SOLIDS = {
    # verts, (unused, kept for reference), vertices per face
    "d20": (_icosa_v, None, 3),
    "d12": (_dodeca_v, None, 5),
    "d6":  (_cube_v, None, 4),
    "d8":  (_octa_v, None, 3),
    "d10": (_trapez10_v, None, 4),
    # A tetrahedron rests on a face with a VERTEX up -- there is no top face, so
    # the 2r derivation has nothing to point at. Modelled anyway, because
    # fitting it still identifies and measures the die; only the top-face
    # prediction is meaningless. See the doc's per-type notes.
    "d4":  (_tetra_v, None, 3),
}

NO_TOP_FACE = {"d4"}


def _face_normal(v, per_face):
    """A normal whose supporting plane holds a whole face, from geometry alone.

    Originally this took the first vertex of the dual solid, on the reasoning
    that an icosahedron's face normals are a dodecahedron's vertices. True in
    principle, but the two standard vertex constructions are duals only up to a
    rotation, so in practice some candidates pointed at a VERTEX instead --
    support count 1 rather than 3. The dice were resting on a corner, which
    changes the silhouette completely and was silently absorbed by whatever was
    fitted downstream.

    Deriving it from the solid itself removes the assumption. At any vertex of
    these solids, any two of its edges lie on a common face, so the plane
    through a vertex and two of its neighbours is a face plane. The support
    count then confirms it.
    """
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    edge = d[d > 1e-9].min()
    centre = v.mean(axis=0)
    for i in range(len(v)):
        nb = np.where((d[i] > 1e-9) & (d[i] < edge * 1.02))[0]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                n = np.cross(v[nb[a]] - v[i], v[nb[b]] - v[i])
                if np.linalg.norm(n) < 1e-9:
                    continue
                u = n / np.linalg.norm(n)
                if u @ (v[i] - centre) < 0:
                    u = -u                       # orient outward
                dd = v @ u
                if int(np.sum(np.abs(dd - dd.max()) < 1e-6)) == per_face:
                    return u
    raise ValueError("no face normal found")


def rest_on_face(kind, inradius):
    """Vertices of `kind` with inradius `inradius`, sitting on a face on z=0."""
    vf, _, per_face = SOLIDS[kind]
    v = vf()
    nrm = _face_normal(v, per_face)
    r0 = float(np.max(v @ nrm))               # supporting plane = inradius
    # rotate that face normal to -z, so the face lies on the floor
    t = np.array([0.0, 0.0, -1.0])
    a = np.cross(nrm, t); s = float(np.linalg.norm(a)); c = float(nrm @ t)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        ax = a / s
        K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        R = np.eye(3) + s*K + (1-c)*(K @ K)
    out = (R @ v.T).T * (float(inradius) / r0)
    out[:, 2] -= out[:, 2].min()              # rest on the floor
    return out

def yaw(v, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return (np.array([[c,-s,0],[s,c,0],[0,0,1.0]]) @ v.T).T

def project(v, centre_xy, rvec, tvec, K):
    w = v.copy(); w[:, 0] += centre_xy[0]; w[:, 1] += centre_xy[1]
    p, _ = cv2.projectPoints(w, rvec, tvec, K, None)
    return p.reshape(-1, 2).astype(np.float32)

def silhouette(v, centre_xy, rvec, tvec, K):
    p = project(v, centre_xy, rvec, tvec, K)
    h = cv2.convexHull(p)
    return float(cv2.contourArea(h)), h
