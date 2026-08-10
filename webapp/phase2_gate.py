"""Phase 2 gate, reading the calibration rather than assuming it."""
import sys, json, requests, numpy as np, cv2
sys.path.insert(0,"."); sys.path.insert(0, r"C:\Claude\smart-table\webapp")
from tray_map import solve_pose
import diemodel as dm

TAG = sys.argv[1]
cal = requests.get("http://10.0.0.23:8081/calibration", timeout=20).json()
if cal.get("warning"):
    sys.exit("calibration is not self-describing: " + cal["warning"])
SQ, ZH, FEAT = cal["square_mm"], cal["height_mm"], cal["feature"]
F = 3547.0
pose = solve_pose(cal["quad"], SQ, cal["frame"], f_px=F, quad_height_mm=ZH)
Rm,_ = cv2.Rodrigues(pose["rvec"]); C = -Rm.T @ pose["tvec"].reshape(3)
Kinv = np.linalg.inv(pose["K"]); S = 4
print(f"calibration: {FEAT} {SQ:.0f} mm at +{ZH:.0f} mm | camera {pose['height_mm']:.0f} mm, "
      f"tilt {pose['tilt_deg']:.1f} deg, reproj {pose['reproj_px']:.1f} px\n")
seg = json.load(open(f"lt_{TAG}.json"))["seg"]

def bp(uv, h):
    d = Rm.T @ (Kinv @ np.array([uv[0], uv[1], 1.0])); return (C+(h-C[2])/d[2]*d)[:2]
def raster(pts):
    m = np.zeros((2592//S, 4608//S), np.uint8); cv2.fillPoly(m,[(pts/S).astype(np.int32)],255); return m
def iou(a,b):
    u = np.count_nonzero(a|b); return np.count_nonzero(a&b)/u if u else 0.0

def fit(die, kind):
    obs = raster(np.array(die["contour"], np.float32))
    oa = cv2.contourArea(np.array(die["contour"], np.float32))
    lo, hi = 2.0, 26.0
    for _ in range(26):
        r = (lo+hi)/2
        v = dm.rest_on_face(kind, r); xy = bp(die["centroid"], r)
        a = np.mean([dm.silhouette(dm.yaw(v,t), xy, pose["rvec"], pose["tvec"],
                                   pose["K"])[0] for t in (0,20,40)])
        if a < oa: lo = r
        else: hi = r
    r = (lo+hi)/2; xy = np.array(bp(die["centroid"], r)); best = (0.0, r, xy, 0.0)
    for it,(dxy,dr,dy) in enumerate(((2.5,0.10,5.0),(1.0,0.04,2.0),(0.4,0.015,0.7))):
        r0, xy0, y0 = best[1], best[2], best[3]
        for rr in (r0*(1-dr), r0, r0*(1+dr)):
            v = dm.rest_on_face(kind, rr)
            for ox in (-dxy,0,dxy):
                for oy in (-dxy,0,dxy):
                    p = xy0+np.array([ox,oy])
                    ys = np.arange(0,120,dy) if it==0 else np.arange(y0-3*dy,y0+3*dy+1e-9,dy)
                    for t in ys:
                        _,h = dm.silhouette(dm.yaw(v,t), p, pose["rvec"], pose["tvec"], pose["K"])
                        sc = iou(obs, raster(h.reshape(-1,2)))
                        if sc > best[0]: best = (sc, rr, p, float(t))
    return best

KINDS = ("d4","d6","d8","d10","d12","d20")
out = []
for d in seg["dice"]:
    if not d["contour"]: continue
    p0 = bp(d["centroid"], 0.0)
    if abs(p0[0]) > SQ/2+6 or abs(p0[1]) > SQ/2+6: continue
    res = {k: fit(d,k) for k in KINDS}
    order = sorted(KINDS, key=lambda k: -res[k][0])
    kind = order[0]; s,r,xy,t = res[kind]
    v = dm.yaw(dm.rest_on_face(kind, r), t)
    if kind in dm.NO_TOP_FACE:
        px, inside = None, None
    else:
        top = v[v[:,2] > v[:,2].max()-r*0.02]
        ctr = top.mean(axis=0) + np.array([xy[0], xy[1], 0.0])
        pr,_ = cv2.projectPoints(ctr.reshape(1,3), pose["rvec"], pose["tvec"], pose["K"], None)
        px = pr.ravel()
        inside = cv2.pointPolygonTest(np.array(d["contour"], np.float32),
                                      (float(px[0]),float(px[1])), False) >= 0
    out.append(dict(id=d["id"], kind=kind, iou=s, margin=s-res[order[1]][0],
                    r=r, yaw=t, runner=order[1],
                    pred=None if px is None else [float(px[0]),float(px[1])],
                    shift=None if px is None else float(np.hypot(*(px-np.array(d["centroid"])))),
                    inside=inside))
json.dump(out, open(f"gate_{TAG}.json","w"))
print(f"{'id':>3} {'best':>5} {'IoU':>6} {'2nd':>5} {'margin':>7} {'2r mm':>7} {'shift':>7} {'in mask':>8}")
for o in out:
    sh = "  n/a" if o["shift"] is None else f"{o['shift']:7.0f}"
    im = "n/a" if o["inside"] is None else ("yes" if o["inside"] else "NO")
    print(f"{o['id']:>3} {o['kind']:>5} {o['iou']:>6.3f} {o['runner']:>5} {o['margin']:>+7.3f} "
          f"{2*o['r']:>7.1f} {sh} {im:>8}")
tf = [o for o in out if o["inside"] is not None]
print(f"\ntop-face centre inside the silhouette: {sum(o['inside'] for o in tf)}/{len(tf)}"
      f"   ({len(out)-len(tf)} d4 excluded -- no top face)")
