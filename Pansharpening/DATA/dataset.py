import torch.utils.data as data
import torch, random
import numpy as np
from os import listdir
from os.path import join
from PIL import Image



class Data(data.Dataset):
    def __init__(self, data_dir,dataset,is_train=False):
        super(Data, self).__init__()
        self.patch_size = 32
        self.upscale_factor = 4
        if is_train:
            data_dir_ms = join(data_dir,dataset+'_data','train128','ms')
            data_dir_pan = join(data_dir, dataset + '_data', 'train128', 'pan')
        else:
            data_dir_ms = join(data_dir, dataset + '_data', 'test128', 'ms')
            data_dir_pan = join(data_dir, dataset + '_data', 'test128', 'pan')

        self.target_list = [Image.open(join(data_dir_ms,i)) for i in listdir(data_dir_ms)]
        self.pan_image = [Image.open(join(data_dir_pan,i)) for i in listdir(data_dir_pan)]
        self.ms_list = [ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)for ms_image in self.target_list]

        self.target_list = [torch.from_numpy(np.array(ms_image).transpose((2, 1, 0))).float() for ms_image in self.target_list]
        self.pan_image = [torch.from_numpy(np.array(ms_image)[np.newaxis,:].transpose((0,2,1))).float() for ms_image in
                            self.pan_image]
        self.ms_list = [torch.from_numpy(np.array(ms_image).transpose((2, 1, 0))).float() for ms_image in
                            self.ms_list]

    def __getitem__(self, index):
        ms_image = self.ms_list[index]
        pan_image = self.pan_image[index]
        target_image = self.target_list[index]
        return ms_image, pan_image, target_image
    def __len__(self):
        return len(self.target_list)


