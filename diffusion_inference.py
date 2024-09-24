import torch
import efficient_unet
import diffusion_dataset
from tqdm import tqdm

class FaceUpsampler():
    def __init__(self, model_path, n_steps=1000):
        self.model = efficient_unet.UNet(6,3).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.n_steps = n_steps
        self.beta = torch.linspace(1e-4, 0.02, self.n_steps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def upsample(self, x):
        with torch.no_grad():
        # x: image, 64x64
        #https://arxiv.org/pdf/2104.07636 pp. 3
            self.model.eval()

            upsampled_img = torch.nn.functional.interpolate(x.unsqueeze(0), scale_factor=4, mode='bicubic')
            noise = torch.randn_like(upsampled_img)
            image = torch.cat([upsampled_img, noise], dim=1)

            for t in tqdm(reversed(range(self.n_steps))):
                noise_hat = self.model(image, torch.Tensor([t]).cuda())
                alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
                alpha_hat = self.alpha_hat[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()

                if t > 1:
                    z = torch.randn_like(noise_hat)
                else:
                    z = torch.zeros_like(noise_hat)

                factor = 1/torch.sqrt(alpha)
                denoise_factor = (1-alpha)/torch.sqrt(1-alpha_hat)
                noising_factor = torch.sqrt(1-alpha)

                image[:, 3:,:,:] = factor * (image[:, 3:,:,:] - (denoise_factor * noise_hat)) + (noising_factor * z)

            image = (image[:, 3:,:,:].clamp(-1, 1) + 1) / 2
            return image
    
torch.cuda.empty_cache()
ds = diffusion_dataset.SuperResDataset('D:/flickr_faces', train=False)
img = ds[1]

upsampler = FaceUpsampler('faces_superres_unet.pth')
upsampled_img = upsampler.upsample(img.cuda()).cpu().squeeze()
print(upsampled_img.size())
import matplotlib.pyplot as plt
import numpy as np

#plot both the lowres and the upsampled image
fig, ax = plt.subplots(1, 2)
lowres = img[:3,:,:]
ax[0].imshow(img.permute(1,2,0))
ax[1].imshow(upsampled_img.permute(1,2,0))
plt.show()

#plt.imshow(upsampled_img.permute(1,2,0))
#plt.show()

