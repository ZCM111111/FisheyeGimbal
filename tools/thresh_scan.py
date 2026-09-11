import glob, os, cv2, numpy as np
FILES = sorted(glob.glob(r"C:\Users\93543\Downloads\IMG_77*.PNG"))

def detect(path, chroma_t, y_t, morph=3, use_bright=False):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    step = max(1, raw.shape[1] // 256)
    small = raw[::step, ::step]
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycc[...,1].astype(np.int16)-128; cb = ycc[...,2].astype(np.int16)-128
    y  = ycc[...,0].astype(np.int16)
    if use_bright:
        # violet OR bright-bloomed button core: chroma low but very bright
        mask = (((np.minimum(cr,cb) > chroma_t) & (y > y_t)) | (y > 235)).astype(np.uint8)
    else:
        mask = ((np.minimum(cr,cb) > chroma_t) & (y > y_t)).astype(np.uint8)
    if morph: mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((morph,morph),np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    pts=[(cen[i][0],cen[i][1]) for i in range(1,n) if 4 <= st[i,cv2.CC_STAT_AREA] <= 1400]
    if len(pts) < 5: return None
    p = np.array(pts, float)
    for _ in range(3):
        x,yy = p[:,0], p[:,1]
        A = np.stack([2*x,2*yy,np.ones_like(x)],1); bv = x*x+yy*yy
        (cx,cy,c),*_ = np.linalg.lstsq(A,bv,rcond=None)
        r = np.sqrt(max(c+cx*cx+cy*cy,1e-6)); d = np.hypot(x-cx,yy-cy)
        if len(p) <= 5: break
        keep = np.abs(d-d.mean()) <= max(0.28*d.mean(),3)
        if keep.all(): break
        p = p[keep]
    x,yy = p[:,0], p[:,1]; d = np.hypot(x-cx,yy-cy)
    spread = float(d.std()/max(d.mean(),1e-6))
    ang = np.sort(np.degrees(np.arctan2(yy-cy,x-cx)))
    gaps = np.diff(np.concatenate([ang,[ang[0]+360]])); cover = 360-float(gaps.max())
    short = float(min(small.shape[:2]))
    return dict(n=len(p), cx=(cx-small.shape[1]*0.5)/short, cy=(cy-small.shape[0]*0.5)/short,
                r=r/short, spread=spread, cover=cover)

# ground truth from HSV (verified: 12/12)
def truth(path):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    step = max(1, raw.shape[1]//256); small = raw[::step,::step]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[...,0],hsv[...,1],hsv[...,2]
    mask = (((h>=122)&(h<=172))&(s>60)&(v>70)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    n,lab,st,cen = cv2.connectedComponentsWithStats(mask,8)
    pts=[(cen[i][0],cen[i][1]) for i in range(1,n) if 4<=st[i,cv2.CC_STAT_AREA]<=1400]
    p=np.array(pts,float)
    for _ in range(3):
        x,yy=p[:,0],p[:,1]; A=np.stack([2*x,2*yy,np.ones_like(x)],1); bv=x*x+yy*yy
        (cx,cy,c),*_=np.linalg.lstsq(A,bv,rcond=None); r=np.sqrt(max(c+cx*cx+cy*cy,1e-6)); d=np.hypot(x-cx,yy-cy)
        if len(p)<=5: break
        keep=np.abs(d-d.mean())<=max(0.28*d.mean(),3)
        if keep.all(): break
        p=p[keep]
    short=float(min(small.shape[:2]))
    return (cx-small.shape[1]*0.5)/short, (cy-small.shape[0]*0.5)/short

T = {os.path.basename(f): truth(f) for f in FILES}

print(f"{'chroma':>7}{'y_min':>7}{'bright':>8}{'pass':>6}{'maxΔcenter':>12}")
print("-"*42)
best=None
for use_bright in (False, True):
    for ct in (4, 6, 8, 10, 12, 14):
        for yt in (50, 70, 90):
            ok=0; worst=0.0; fail=[]
            for f in FILES:
                r = detect(f, ct, yt, use_bright=use_bright)
                name=os.path.basename(f)
                if r and r['spread']<=0.28 and r['cover']>=200:
                    dx=abs(r['cx']-T[name][0]); dy=abs(r['cy']-T[name][1])
                    dev=max(dx,dy); worst=max(worst,dev)
                    ok+=1
                else:
                    fail.append(name[-8:-4])
            print(f"{ct:>7}{yt:>7}{str(use_bright):>8}{ok:>6}{worst:>12.3f}  {' '.join(fail)}")
            if ok==12 and (best is None or worst<best[0]):
                best=(worst,ct,yt,use_bright)
print("-"*42)
print("最佳:", best)
