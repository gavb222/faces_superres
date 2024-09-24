#Faces-Superres
This project is a diffusion model based on 'Image Super-Resolution via Iterative Refinement' (https://arxiv.org/pdf/2104.07636), and trained on the Flickr Faces dataset, containing 52,000 faces. It can be found at https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq.

I use a modified version of efficient-unet as detailed in Google Deepmind's Imagen research, modified to fit the model on my local GPU, as well as a modified diffusion timestep calculating system.

I use the super-resolution paper's technique of conditioning on the low-resolution image by concatenating a bicubically upsampled version of the low resolution image to the noised ground truth image in the channel dimension during training.

A few photos of results are shown below:
