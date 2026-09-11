import os
import sys

# Frame folders are passed on the command line so this file carries no
# personal paths: python ycc_test.py <folder> [folder ...]
DATA_DIRS = sys.argv[1:]
if not DATA_DIRS:
    print("usage: python ycc_test.py <frames folder> [more folders]")
    sys.exit(1)

import glob, os, cv2, numpy as np

FILES = sorted(sum([glob.glob(os.path.join(d, "*")) for d in DATA_DIRS], []))

def run(path, use_ycc):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    w = raw.shape[1]; step = max(1, w // 256)
    small = raw[::step, ::step]
    if use_ycc:
        ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        cr = ycc[...,1].astype(np.int16) - 128
        cb = ycc[...,2].astype(np.int16) - 128
        y  = ycc[...,0].astype(np.int16)
        # purple = both chroma channels clearly positive (red + blue, little green)
        score = np.minimum(cr, cb)
        mask = ((score > 14) & (y > 70)).astype(np.uint8)
    else:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
        mask = (((h>=122)&(h<=172))&(s>60)&(v>70)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    pts=[]
    for i in range(1,n):
        a = st[i, cv2.CC_STAT_AREA]
        if 4 <= a <= 1400:
            pts.append((cen[i][0], cen[i][1]))
    if len(pts) < 5: return None
    p = np.array(pts, float)
    for _ in range(3):
        x,y = p[:,0], p[:,1]
        A = np.stack([2*x, 2*y, np.ones_like(x)],1); bv = x*x+y*y
        (cx,cy,c),*_ = np.linalg.lstsq(A,bv,rcond=None)
        r = np.sqrt(max(c+cx*cx+cy*cy,1e-6))
        d = np.hypot(x-cx,y-cy)
        if len(p) <= 5: break
        keep = np.abs(d-d.mean()) <= max(0.28*d.mean(),3)
        if keep.all(): break
        p = p[keep]
    x,y = p[:,0], p[:,1]; d = np.hypot(x-cx,y-cy)
    spread = float(d.std()/max(d.mean(),1e-6))
    ang = np.sort(np.degrees(np.arctan2(y-cy,x-cx)))
    gaps = np.diff(np.concatenate([ang,[ang[0]+360]])); cover = 360-float(gaps.max())
    short = float(min(small.shape[:2]))
    return len(p), (cx-small.shape[1]*0.5)/short, (cy-small.shape[0]*0.5)/short, r/short, spread, cover

print(f"{'file':<14}{'HSV n/cx/cy/r/sp/cov':<40}{'YCbCr n/cx/cy/r/sp/cov':<40}")
print("-"*94)
okH=okY=0
for f in FILES:
    a = run(f, False); b = run(f, True)
    fa = " ".join(f"{v:.2f}" for v in a[1:]) if a else "FAIL"
    fb = " ".join(f"{v:.2f}" for v in b[1:]) if b else "FAIL"
    good_a = a and a[4] <= 0.28 and a[5] >= 200
    good_b = b and b[4] <= 0.28 and b[5] >= 200
    okH += 1 if good_a else 0; okY += 1 if good_b else 0
    print(f"{os.path.basename(f):<14}{str(a[0])+' '+fa:<40}{str(b[0])+' '+fb:<40}")
print("-"*94)
print(f"HSV 通过 {okH}/12    YCbCr 通过 {okY}/12")
