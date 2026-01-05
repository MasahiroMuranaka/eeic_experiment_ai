from depth_anything_v2.dpt import abs_DepthAnythingV2_softplus
import cv2
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'
## pth_name: 重みデータのパス
pth_name = 'model_fold3.pth'
model = abs_DepthAnythingV2_softplus(**model_configs[encoder])
model.load_state_dict(torch.load(pth_name, map_location='cpu'))
model = model.to(DEVICE).eval()

## filename = 画像のパス
filename = "./OIP.png"
## raw_image = ndarray(H, W, 3)
raw_image = cv2.imread(filename)
with torch.no_grad():
    ## infer_image: ndarray(H, W, 3)の画像データを引数
    ## 返り値はdepth = ndarray(H, W)
    depth = model.infer_image(raw_image)
    np.save("./depth.npy", depth)