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

SOLIDS = {
    # verts, and the dual whose vertices give this solid's face normals
    "d20": (_icosa_v, _dodeca_v),
    "d12": (_dodeca_v, _icosa_v),
}

def rest_on_face(kind, inradius):
    """Vertices of `kind` with inradius `inradius`, sitting on a face on z=0."""
    vf, nf = SOLIDS[kind]
    v, n = vf(), nf()
    nrm = n[0] / np.linalg.norm(n[0])
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
