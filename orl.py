import torch.utils.data as data
from PIL import Image

class ORL(data.Dataset):  
    def __init__(self, datatxt, transform=None):
        f = open(datatxt, 'r')
        datalist = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            datalist.append((words[0], int(words[1])))
        
        self.datalist = datalist
        self.transform = transform
        
    def __getitem__(self, index):
        imdir, label = self.datalist[index]
        img = Image.open(imdir)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.datalist)