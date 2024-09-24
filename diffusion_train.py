import torch
#from torch.optim import lr_scheduler
import efficient_unet
import diffusion_dataset
from tqdm import tqdm

torch.cuda.empty_cache()

n_epochs = 100
batch_size = 8
dataset_path = "D:/flickr_faces"

train_dataset = diffusion_dataset.SuperResDataset(dataset_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

val_dataset_path = "D:/CelebA_HQ_resized"
val_dataset = diffusion_dataset.SuperResDataset(val_dataset_path, train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

ckpt_path = 'faces_superres_unet.pth'

model = efficient_unet.UNet(6,3).cuda()
if ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
loss_fn = torch.nn.MSELoss()

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        model_in, noise, noise_timestamp = data
        noise_pred = model(model_in.cuda(), e=noise_timestamp.cuda())
        loss = loss_fn(noise_pred, noise.cuda())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {running_loss/100}")
            running_loss = 0.0
            torch.save(model.state_dict(), 'faces_superres_unet.pth')

    #validation step
    model.eval()
    val_loss = 0.0
    for i, data in tqdm(enumerate(val_loader)):
        model_in, noise, noise_timestamp = data
        noise_pred = model(model_in.cuda(), e=noise_timestamp.cuda())
        loss = loss_fn(noise_pred, noise.cuda())
        val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")
