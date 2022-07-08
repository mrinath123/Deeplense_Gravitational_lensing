import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

def make_labels(paths,target):
    xx = np.array(paths)
    xx = np.expand_dims(xx,1)
    yy = np.zeros((len(xx),1)) + target
    gg = np.concatenate([xx,yy],axis = 1)
    return gg

def prep_data(class1 , class2):
    # class1 is non_lensing images
    # class2 is lensed images
    d1 = make_labels(class1 , 0)
    d2 = make_labels(class2 , 1)
    t_data = np.concatenate([d1,d2] , axis = 0)
    X, X_test = train_test_split(t_data, test_size=0.1, random_state=42) # 10% set to test data
    X_train, X_val = train_test_split(X, test_size=0.25, random_state=42)

    return X_train, X_val , X_test

class Len(Dataset):
    def __init__(self , data , augs):
        self.data = data
        self.augs = augs
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self , idx):
        path = self.data[idx][0] 
        target = float(self.data[idx][1])

        image = np.load(path)      
        image = (image - np.min(image))/(np.max(image) - np.min(image)) #make the values raof image range from 0 to1
        image = np.expand_dims(image , axis = 2)
        
        transformed = self.augs(image=image)       
        image = transformed['image']
        image = torch.tensor(image,dtype = torch.float32)      
         
        return image,torch.tensor(target).long()
