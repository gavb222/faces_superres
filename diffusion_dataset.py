import torch
import os
from PIL import Image

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_lowres = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def noise_images(x, t, beta_start=1e-4, beta_end=0.02, noise_steps=1000):
    "Add noise to image x at time t"
    beta = torch.linspace(beta_start, beta_end, noise_steps)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    eps = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

class SuperResDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, train=True):
        self.data_folder = data_folder
        self.filenames = os.listdir(self.data_folder)
        self.noise_steps = 1000
        self.training = train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        img = Image.open(os.path.join(self.data_folder, file))
        img_highres = preprocess(img)
        img_lowres = preprocess_lowres(img)
        noise_timestamp = torch.randint(low=1, high=self.noise_steps, size=(1,))
        img_upsampled = torch.nn.functional.interpolate(img_lowres.unsqueeze(0), scale_factor=4, mode='bicubic')
        noised_img, noise = noise_images(img_highres, noise_timestamp)
        model_in = torch.cat([img_upsampled, noised_img], dim=1)
        if self.training:
            return model_in.squeeze(), noise, torch.Tensor(noise_timestamp)
        else:
            return img_lowres
    
#ds = SuperResDataset('D:/flickr_faces')

#visualize the images
#import matplotlib.pyplot as plt
#import numpy as np
#fig, ax = plt.subplots(1, 2)
#img, noise, noise_timestamp = ds[0]
#print(noise_timestamp)
#lowres = img[:3,:,:]
#noised = img[3:,:,:]
#ax[0].imshow(lowres.permute(1,2,0))
#ax[0].set_title("Lowres")
#ax[1].imshow(noised.permute(1,2,0))
#ax[1].set_title("Noised")
#plt.show()



