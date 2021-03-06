from PIL import Image 

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os


path = '/home/aims/Documents/Pytorch/pytorch_exercise/data'

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

image_size     = (100, 100)
image_row_size = image_size[0] * image_size[1]
image_size     = (100, 100)
image_row_size = image_size[0] * image_size[1]


transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                # transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])

class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes   = os.listdir(path)
        self.path = [f"{path}/{className}" for className in self.classes]
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform
        
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
        self.file_list = files
        files = None

        
    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory
      
