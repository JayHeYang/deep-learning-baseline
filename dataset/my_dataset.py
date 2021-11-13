def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import KFold
from PIL import Image
import numpy as np


class Cifar10_Dataset(data.Dataset):
    def __init__(self, data_roots, trans):
        self.transforms = trans
        for k, data_root in enumerate(data_roots):
            data = unpickle(data_root)[b'data']
            label = unpickle(data_root)[b'labels']
            if k == 0:
                self.all_data = data
                self.all_label = label
            else:
                self.all_data = np.vstack((self.all_data, data)) # shape->(50000, 3072) or (10000, 3072)
                self.all_label = self.all_label + label # shape->(50000, ) or (10000, )
        

    def __getitem__(self, idx):
        # 调整数据形状为图片格式 (3x32x32), 之后调整通道和图片宽高的排序（CHW->HWC）
        sample = np.reshape(self.all_data[idx], (3, 32, 32)).transpose((1, 2, 0))
        img = Image.fromarray(sample)
        img = self.transforms(img) # 3x32x32
        return img, self.all_label[idx]

    def __len__(self):
        return len(self.all_label)


# if __name__ == '__main__':
#     # storage location datasets
#     file = '/Users/morvan/Downloads/cifar-10-batches-py/data_batch_1'
#     roots = [file] 
#     trans = T.Compose([
#         T.RandomCrop(32),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     dataset = Cifar10_Dataset(roots, trans)
#     print(dataset.__getitem__(10)[0].shape)
    