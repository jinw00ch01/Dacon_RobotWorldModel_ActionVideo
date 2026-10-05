import glob, os, numpy as np, cv2
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'open', 'data', 'train')
OUT = os.path.dirname(__file__)
dss = sorted(glob.glob(os.path.join(R, '*', '*', 'meta')))
ims = []
with open(os.path.join(OUT, 'train_grid_idx.txt'), 'w', encoding='utf-8') as fo:
    for i, m in enumerate(dss):
        v = sorted(glob.glob(os.path.join(os.path.dirname(m), 'videos', '*', '*', '*.mp4')))[0]
        cap = cv2.VideoCapture(v); ok, fr = cap.read()
        fr = cv2.resize(fr if ok else np.zeros((480, 640, 3), np.uint8), (160, 120))
        cv2.putText(fr, str(i), (3, 14), 0, 0.5, (0, 255, 255), 1); ims.append(fr)
        fo.write(f"{i} {os.path.relpath(os.path.dirname(m), R)}\n")
while len(ims) % 16: ims.append(np.zeros_like(ims[0]))
g = np.vstack([np.hstack(ims[i:i + 16]) for i in range(0, len(ims), 16)])
cv2.imwrite(os.path.join(OUT, 'train_grid.jpg'), g)

