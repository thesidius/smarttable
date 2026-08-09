"""Phase 2 gate: predict each die's TOP FACE CENTRE and see where it lands."""
import sys, json, requests, numpy as np, cv2
sys.path.insert(0,"."); sys.path.insert(0, r"C:\Claude\smart-table\webapp")
from tray_map import solve_pose
import diemodel as dm

SQ, F = 93.0, 3547.0
cal = requests.get("http://10.0.0.23:8081/calibration", timeout=20).json()
quad, frame = cal["quad"], cal["frame"]
pose = solve_pose(quad, SQ, frame, f_px=F)
Rm,_ = cv2.Rodrigues(pose["rvec"]); C = -Rm.T @ pose["tvec"].reshape(3)
Kinv = np.linalg.inv(pose["K"])
seg = json.load(open("lt_FLOOR.json"))["seg"]
S = 4                                            # IoU raster downscale

def bp(uv, h):
    d = Rm.T @ (Kinv @ np.array([uv[0], uv[1], 1.0]))
    return (C + (h-C[2])/d[2]*d)[:2]

def raster(pts):
    m = np.zeros((2592//S, 4608//S), np.uint8)
    cv2.fillPoly(m, [(pts/S).astype(np.int32)], 255)
    return m

def iou(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0

def fit(die, kind):
    obs = raster(np.array(die["contour"], np.float32))
    oa = cv2.contourArea(np.array(die["contour"], np.float32))
    lo, hi = 4.0, 30.0                            # seed r from area
    for _ in range(28):
        r = (lo+hi)/2
        v = dm.rest_on_face(kind, r); xy = bp(die["centroid"], r)
        a = np.mean([dm.silhouette(dm.yaw(v,t), xy, pose["rvec"], pose["tvec"],
                                   pose["K"])[0] for t in (0,20,40)])
        if a < oa: lo = r
        else: hi = r
    r = (lo+hi)/2; xy = np.array(bp(die["centroid"], r)); best = (0.0, r, xy, 0.0)
    span = 120.0 if kind == "d12" else 120.0
    for it, (dxy, dr, dyaw) in enumerate(((3.0, 0.10, 4.0), (1.0, 0.04, 1.5), (0.4, 0.015, 0.6))):
        r0, xy0, y0 = best[1], best[2], best[3]
        for rr in (r0*(1-dr), r0, r0*(1+dr)):
            v = dm.rest_on_face(kind, rr)
            for ox in (-dxy, 0, dxy):
                for oy in (-dxy, 0, dxy):
                    p = xy0 + np.array([ox, oy])
                    ys = np.arange(0, span, dyaw) if it == 0 else \
                         np.arange(y0-3*dyaw, y0+3*dyaw+1e-9, dyaw)
                    for t in ys:
                        _, h = dm.silhouette(dm.yaw(v,t), p, pose["rvec"],
                                             pose["tvec"], pose["K"])
                        s = iou(obs, raster(h.reshape(-1,2)))
                        if s > best[0]: best = (s, rr, p, float(t))
    return best

out = []
for d in seg["dice"]:
    if not d["contour"]:
        continue
    p0 = bp(d["centroid"], 0.0)
    if abs(p0[0]) > SQ/2+8 or abs(p0[1]) > SQ/2+8:
        continue
    cands = {k: fit(d, k) for k in ("d20", "d12")}
    kind = max(cands, key=lambda k: cands[k][0])
    s, r, xy, t = cands[kind]
    v = dm.yaw(dm.rest_on_face(kind, r), t)
    top = v[v[:,2] > v[:,2].max() - r*0.02]
    ctr3 = top.mean(axis=0) + np.array([xy[0], xy[1], 0.0])
    px, _ = cv2.projectPoints(ctr3.reshape(1,3), pose["rvec"], pose["tvec"], pose["K"], None)
    px = px.ravel()
    cen = np.array(d["centroid"])
    inside = cv2.pointPolygonTest(np.array(d["contour"], np.float32),
                                  (float(px[0]), float(px[1])), False) >= 0
    out.append(dict(id=d["id"], kind=kind, iou=s, r=r, yaw=t,
                    pred=[float(px[0]), float(px[1])],
                    shift=float(np.hypot(*(px-cen))), inside=bool(inside),
                    other=cands["d12" if kind=="d20" else "d20"][0]))
json.dump(out, open("gate.json","w"))
print(f"{'id':>3} {'type':>5} {'IoU':>6} {'margin':>7} {'2r mm':>7} {'offset px':>10} {'in mask':>8}")
for o in out:
    print(f"{o['id']:>3} {o['kind']:>5} {o['iou']:>6.3f} {o['iou']-o['other']:>+7.3f} "
          f"{2*o['r']:>7.1f} {o['shift']:>10.1f} {'yes' if o['inside'] else 'NO':>8}")
print(f"\npredicted centre inside the silhouette: {sum(o['inside'] for o in out)}/{len(out)}")
print(f"median offset from the centroid: {np.median([o['shift'] for o in out]):.0f} px "
      f"-- this is the correction the doc is about; zero would mean it does nothing")
