from torch.utils.data import Dataset
import PIL.Image as Image
import os

EXTEND_IMG = 6

def make_dataset_(root):
    imgs=[]
    n=len(os.listdir(root))
    a = n // EXTEND_IMG
    for i in range(1, a+1):
        for j in range(EXTEND_IMG):
            img=os.path.join(root,"history%d%d.jpg"%(i,j))
            imgs.append(img)
    return imgs

class RacingCarDataset(Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset_(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path)#.convert('L')

        # img_x = img_x.resize((96, 96))
        # img_x.save(x_path)

        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x

    def __len__(self):
        return len(self.imgs)
