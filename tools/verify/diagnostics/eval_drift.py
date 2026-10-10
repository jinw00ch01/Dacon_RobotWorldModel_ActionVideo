"""Diagnosis only (not for selection or tuning): label-free statistics of submitted eval videos.

Per clip: DINO cosine distance of each generated frame to the input frame (scene drift), mean over t=1..15 and at t=15.
"""
import glob, os, sys
import av, cv2, numpy as np, pandas as pd, timm, torch
sys.path.insert(0, r'C:\Dacon\RobotWorldModel_ActionVideo\work\agents\verifier')
from tools.verify.feature_scores import letterbox
torch.set_num_threads(8)
m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).eval()
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
S = r'C:\Dacon\WM_Shared\submissions'
subs = {'sub2': 'v2_s8000_g3', 'sub3': 'sub3', 'sub4': 'sub4', 'sub5': 'sub5', 'sub7': 'sub7'}
rows = []
for k, d in subs.items():
    for p in sorted(glob.glob(os.path.join(S, d, 'anchor_t20', 'videos', '*.mp4'))):
        with av.open(p) as c:
            fr = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
        x = np.stack([cv2.resize(letterbox(letterbox(f), 224, 224), (224, 224)) for f in fr]).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD).astype(np.float32)).permute(0, 3, 1, 2)
        with torch.no_grad():
            f = torch.nn.functional.normalize(m(x), dim=1).numpy()
        dist = 1 - f[1:] @ f[0]
        sid = os.path.basename(p)[:-4]
        rows.append(dict(sub=k, sample_id=sid, scene=0 if int(sid[-6:]) <= 153 else 1, drift_mean=dist.mean(), drift_end=dist[-1]))
    print(k, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(r'C:\Dacon\WM_Shared\verify_qa\eval_submission_drift_diagnostic.csv', index=False)
print(df.groupby(['sub', 'scene'])[['drift_mean', 'drift_end']].mean().round(4).unstack())
print(df.groupby('sub')[['drift_mean', 'drift_end']].mean().round(4))
