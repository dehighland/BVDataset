import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_link = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_link)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Create reference variables for given row of dataset
        name = np.array([int(self.data_link.iloc[idx, 0][self.data_link.iloc[idx, 0].index('pt') + 2: self.data_link.iloc[idx, 0].index(' (')])]) # Grabs pt number
        image = Image.open(self.data_link.iloc[idx, 0]).convert('RGB')
        pH = np.array([self.data_link.iloc[idx, 1]]) 
        Clue = np.array([self.data_link.iloc[idx, 2]])  
        Mol = np.array([self.data_link.iloc[idx, 3]])
        apH = np.array([self.data_link.iloc[idx, 4]])
        whiff = np.array([self.data_link.iloc[idx, 5]])
        diag = np.array([self.data_link.iloc[idx, 6]])

        # Piece together how to access a data object by label name
        sample = {"Image": image, "pH": pH, "ClueCell": Clue, "Molecular": Mol, "Adj_pH": apH, "Whiff": whiff, "Diagnosis": diag, "Name": name}

        # Applying transforms to images
        if self.transform:
            sample["Image"] = self.transform(sample["Image"])

        return sample
 
def load_data(batch_size, img_size):
    '''Main function to create dataloaders for model training and testing using passed batch size and common image size'''
    
    # Data Pre-processing and augmentation
    train_trans = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((img_size,img_size)), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    test_trans = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((img_size,img_size)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
                                    
    # Load train and testing datasets
    train_data = CustomDataset("./Data/training/Labels.csv", train_trans)
    test_data = CustomDataset("./Data/testing/Labels.csv", test_trans)
                                                    
    # DataLoaders
    train_DL = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_DL = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print('Dataloaders produced!')
    return train_DL, test_DL
    