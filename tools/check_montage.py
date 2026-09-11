import glob, os, cv2, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("st", r"C:\Users\93543\Desktop\SteadyFisheye\tools\struct_test.py")
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)

arc = sorted(glob.glob(r"D:\桌面文件\训练2\*.png"))[:2] + sorted(glob.glob(r"D:\桌面文件\训练数据\*.png"))[:2]
home = sorted(glob.glob(r"C:\Users\93543\Downloads\IMG_77*.PNG"))[:2]
files = arc + home
tiles = []
for f in files:
    r = st.analyse(f)
    if r is None: continue
    vis = r["small"].copy()
    if vis.shape[1] > 420:
        k = 420.0/vis.shape[1]
        vis = cv2.resize(vis, None, fx=k, fy=k)
    else:
        k = 1.0
    ok = r.get("ok")
    col = (0,255,0) if ok else (0,0,255)
    cv2.circle(vis, (int(r["cx"]*k), int(r["cy"]*k)), int(r["r"]*k), col, 2)
    cv2.drawMarker(vis, (int(r["cx"]*k), int(r["cy"]*k)), col, cv2.MARKER_CROSS, 20, 2)
    txt = f"{os.path.basename(f)[:18]} {'OK' if ok else 'FAIL'} sup={r.get('support',0):.2f} r={r['rNorm']:.2f}"
    cv2.rectangle(vis, (0,0), (vis.shape[1], 22), (0,0,0), -1)
    cv2.putText(vis, txt, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    tiles.append(vis)
h = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, h-t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20)) for t in tiles]
row1 = np.hstack(tiles[:3]); row2 = np.hstack(tiles[3:])
w = max(row1.shape[1], row2.shape[1])
row1 = cv2.copyMakeBorder(row1,0,0,0,w-row1.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20))
row2 = cv2.copyMakeBorder(row2,0,0,0,w-row2.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20))
out = np.vstack([row1,row2])
cv2.imwrite(r"C:\Users\93543\Desktop\SteadyFisheye\tools\check_montage.jpg", out)
print("montage:", out.shape)
for f in files:
    r = st.analyse(f)
    if r: print(f"{os.path.basename(f):<26} ok={r.get('ok')} method={r['method']} sup={r['support']:.2f} rN={r['rNorm']:.3f} c=({r['cxNorm']:+.3f},{r['cyNorm']:+.3f})")
