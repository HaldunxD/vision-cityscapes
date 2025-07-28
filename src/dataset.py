import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pathlib import Path
import cv2, json, numpy as np

CITY_COLORS = json.load(open(Path(__file__).parent / "city_colors.json"))

class CityscapesDataset(Dataset):
    def __init__(self, root, split="train", crop_size=(512,1024)):
        img_dir = Path(root)/"leftImg8bit"/split
        mask_dir = Path(root)/"gtFine"/split
        self.samples = sorted(img_dir.rglob("*_leftImg8bit.png"))
        self.mask_names = [Path(str(p).replace("leftImg8bit","gtFine")
                                         .replace("_leftImg8bit","_gtFine_labelIds")) for p in self.samples]

        self.tfm = A.Compose([
            A.RandomCrop(*crop_size),
            A.HorizontalFlip(p=.5),
            A.ColorJitter(.2,.2,.2,.1,p=.5),
            A.Normalize(mean=(0.286,0.325,0.283), std=(0.186,0.190,0.187)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.samples[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_names[idx]), 0)
        aug = self.tfm(image=img, mask=mask)
        return aug["image"], aug["mask"].long()
