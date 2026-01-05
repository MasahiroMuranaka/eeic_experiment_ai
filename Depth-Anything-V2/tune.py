import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torchvision.transforms as T
import os
from depth_anything_v2.dpt import abs_DepthAnythingV2_softplus
import cv2
import numpy as np

# ----- Dataset -----
class DepthDataset(Dataset):
    def __init__(self, images, depths, transform=None):
        self.images = images
        self.depths = depths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        depth = self.depths[idx]

        if self.transform:
            img, depth = self.transform(img, depth)
        return img, depth

# ----- 水平反転 -----
class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth):
        if np.random.rand() < self.p:
            img = np.flip(img, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
        return img, depth

# ----- 学習関数 -----
def train_fold(model, dataset, train_idx, val_idx, device, num_epochs=20, save_path=None):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # optimizer: encoder は凍結、decoder と scale head を学習
    params = list(model.depth_head.parameters()) + list(model.add_layer.parameters())
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    criterion = nn.L1Loss()

    model.to(device)

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        model.train()
        sum_train_loss = 0.0
        for imgs, depths in train_loader:
            imgs = imgs[0]
            depths = torch.from_numpy(depths).unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model.infer_image_torch(imgs)
            train_loss = criterion(outputs, depths)
            sum_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
        
        ave_train_loss = sum_train_loss / len(train_loader.dataset)

        model.eval()
        sum_val_loss = 0.0
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs = imgs[0]
                depths = torch.from_numpy(depths).unsqueeze(1).to(device)
                outputs = model.infer_image_torch(imgs)
                sum_val_loss += criterion(outputs, depths).item()
        
        ave_val_loss = sum_val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}, TrainLoss: {ave_train_loss.item():.4f}, ValLoss: {ave_val_loss.item():.4f}")

    # 学習後に重みを保存
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

# ----- k-fold -----
def k_fold_training(images, depths, k=5, device='cuda', save_dir='./checkpoints'):
    transform = HorizontalFlip(p=0.5)
    dataset = DepthDataset(images, depths, transform=transform)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits' # or 'vits', 'vitb', 'vitg'
    model = abs_DepthAnythingV2_softplus(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'), strict=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"=== Fold {fold+1} ===")
        # encoder を凍結
        for param in model.pretrained.parameters():
            param.requires_grad = False

        save_path = os.path.join(save_dir, f'model_fold{fold+1}.pth')
        train_fold(model, dataset, train_idx, val_idx, device, save_path=save_path)

if __name__ == '__main__':
    images = []
    depths = []
    for i in range(1449):
        img_filename = f'./data/images/image_{i:04d}.png'
        dpt_filename = f'./data/depths/depth_{i:04d}.npy'

        images.append(cv2.imread(img_filename))
        depths.append(np.load(dpt_filename))
    
    save_dir = "./checkpoints"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    k_fold_training(images=images, depths=depths, device=device, save_dir=save_dir)