from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.transforms import add_new_channels

class CIFAKEDataset(Dataset):
    def __init__(self, dataset_path, split, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        assert split in {"train", "test"}, f"got {split}"
        self.split = split
        
        self.items = self.parse_dataset()
        
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_dataset(self):
        def is_image(filename):
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    return True
            return False
        
        split_path = join(self.dataset_path, self.split)
        items = [{
            "image_path":  join(split_path, "REAL", image_path),
            "is_real": True
            } for image_path in listdir(join(split_path, "REAL")) if is_image(image_path)] + [{
            "image_path":  join(split_path, "FAKE", image_path),
            "is_real": False
            } for image_path in listdir(join(split_path, "FAKE")) if is_image(image_path)]
        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Lambda(add_new_channels),
        ])(image)

        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"]),
            "is_real": torch.as_tensor([1 if self.items[i]["is_real"] is True else 0]),
        }
        return sample