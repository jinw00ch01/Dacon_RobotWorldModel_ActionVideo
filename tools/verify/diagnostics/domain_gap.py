import glob, json, os, sys
import cv2, numpy as np, pandas as pd, polars as pl, timm, torch
sys.path.insert(0, r'C:\Dacon\RobotWorldModel_ActionVideo\work\agents\verifier')
from tools.verify.feature_scores import letterbox
R = r'C:\Dacon\RobotWorldModel_ActionVideo\work\agents\verifier'
torch.set_num_threads(8)
m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).eval()
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def emb(imgs_rgb):
    out = []
    for i in range(0, len(imgs_rgb), 32):
        x = np.stack([cv2.resize(letterbox(im, 224, 224), (224, 224)) for im in imgs_rgb[i:i + 32]]).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD).astype(np.float32)).permute(0, 3, 1, 2)
        with torch.no_grad():
            f = m(x)
        out.append(torch.nn.functional.normalize(f, dim=1).numpy())
    return np.concatenate(out)

def first_frame(path):
    cap = cv2.VideoCapture(path); ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

spl = json.load(open(os.path.join(R, 'configs/splits/holdout_v1.json'), encoding='utf-8'))
eps = pd.read_parquet(r'C:\Dacon\WM_Shared\data_index\episodes.parquet')
eps['key'] = eps.user + '/' + eps.dataset
tr = eps[eps.key.isin(set(spl['train_datasets']))].groupby('key').head(5)
train_imgs = [first_frame(os.path.join(R, 'open/data/train', v)) for v in tr.video]
hold = sorted(glob.glob(r'C:\Dacon\WM_Shared\holdout_v1\images\*.png'))
ev = sorted(glob.glob(os.path.join(R, 'open/data/eval/images/*.png')))
rd = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
T, H, E = emb(train_imgs), emb([rd(p) for p in hold]), emb([rd(p) for p in ev])
nn = lambda X: 1 - (X @ T.T).max(1)
dh, de = nn(H), nn(E)
win = pd.read_csv(r'C:\Dacon\WM_Shared\holdout_v1\windows.csv')
hdf = pd.DataFrame({'sample_id': [os.path.basename(p)[:-4] for p in hold], 'nn_dist': dh}).merge(win[['sample_id', 'user']], on='sample_id')
hdf.to_csv(r'C:\Dacon\WM_Shared\verify_scores\holdout192_nn_dist_to_train.csv', index=False)
print('train imgs', len(T))
print('holdout NN cos-dist to train: median %.3f  [p10 %.3f, p90 %.3f]' % (np.median(dh), *np.percentile(dh, [10, 90])))
print(hdf.groupby('user').nn_dist.median().round(3).to_dict())
print('eval scene0 median %.3f, scene1 median %.3f' % (np.median(de[:154]), np.median(de[154:])))
print('fraction of eval beyond holdout p90: scene0 %.2f scene1 %.2f' % ((de[:154] > np.percentile(dh, 90)).mean(), (de[154:] > np.percentile(dh, 90)).mean()))
