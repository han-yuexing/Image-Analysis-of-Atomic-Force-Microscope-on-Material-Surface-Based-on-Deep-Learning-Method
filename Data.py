import random
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torch

class Tai(data.Dataset):
    def __init__(self, X_files, Y_files=None, validate_indexes=None, validate=False, test=False):
        #0-266889.png
        self.X_files = X_files
        self.Y_files = Y_files
        self.Label = None
        if self.Y_files is not None:
            self.Label = np.load(self.Y_files)
            self.Label = np.array(self.Label)
        self.validate_indexes = validate_indexes
        self.validate = validate
        self.test = test
        if not self.validate and not self.test:
            self.train = np.delete(np.array(range(0, 266889)), np.array(self.validate_indexes))
        self.class_names = np.array([
            'background',
            'target'
            ])
    
    def __len__(self):
        if self.test:
            return 307200
        if self.validate:
            return len(self.validate_indexes)
        return 250150 - len(self.validate_indexes)

   
    def __getitem__(self, idx):
        if self.validate:
            real_index = self.validate_indexes[idx]
        else:
            if self.test:
                real_index = idx
            else:
                real_index = self.train[idx]
        if self.Label is not None:
            label = self.Label[real_index]
        img_path = os.path.join(self.X_files, str(real_index)+".png")
        img = np.array(Image.open(img_path), dtype='float32')[:,:,0]
        img = torch.tensor(img).unsqueeze(dim=0)
        if self.test:
            return img
        return img, label

if __name__ == "__main__":
    validate_indexes = random.sample(range(0, 266889), 20000)
    TrainSet = Tai(X_files = "./Data", Y_files = "./Label.npy", validate_indexes = validate_indexes)
    print(len(TrainSet))
    ValidSet = Tai(X_files = "./Data", Y_files = "./Label.npy", validate_indexes = validate_indexes, validate = True)
    print(len(ValidSet))
    train_loader = data.DataLoader(TrainSet, batch_size=32, shuffle=False)
    Valid_loader = data.DataLoader(ValidSet, batch_size=32, shuffle=False)            
