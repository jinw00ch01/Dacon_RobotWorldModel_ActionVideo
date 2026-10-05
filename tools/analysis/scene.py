import os, glob, json, numpy as np, cv2
np.set_printoptions(precision=2, suppress=True, linewidth=200)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'open', 'data')
fs = sorted(glob.glob(os.path.join(R, 'eval', 'images', '*.png')))
feat = np.array([cv2.resize(cv2.imread(f), (4, 3), interpolation=cv2.INTER_AREA).reshape(-1) for f in fs], float)
# 2-means
c = feat[[0, -1]].copy()
for _ in range(20):
    lab = np.argmin(((feat[:, None] - c[None]) ** 2).sum(-1), 1)
    c = np.stack([feat[lab == k].mean(0) for k in range(2)])
print('cluster sizes', np.bincount(lab), 'first idx of cluster1', np.where(lab == 1)[0][:3], 'cluster0 after:', np.where(lab == 0)[0].max())
E = np.stack([np.load(f.replace('images', 'actions').replace('.png', '.npy')) for f in fs])
st = json.load(open(os.path.join(R, 'train', 'so100_action_statistics.json')))
mu, sd = np.array(st['mean']), np.array(st['std'])
Z = (E - mu) / sd
for k in range(2):
    z = Z[lab == k]
    print(f'scene{k} n={len(z)} z-mean', z.reshape(-1, 6).mean(0), 'z-std', z.reshape(-1, 6).std(0))
    print(f'  mean |z_t - z_0| over t (motion in z units)', np.abs(z - z[:, :1]).mean((0, 1)), 'avg', np.abs(z - z[:, :1]).mean())
    print('  gripper range', E[lab == k][..., 5].min(), E[lab == k][..., 5].max())
# naive "action = a0 constant" would give MAE equal to mean |z_t - z_0|; mean |z - mean_train| = MAE of all-zero prediction
print('MAE if extractor predicted a0 for every frame:', np.abs(Z - Z[:, :1]).mean())
print('MAE if extractor predicted train mean (z=0):', np.abs(Z).mean())


