import numpy as np
import torch
import torchvision
from BYOL_model import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE_TRAIN = 64

# Using CIFAR10 as ImageNet is too large
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('./datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)

VIEW_SIZE = 224

# default values for scale, size, ratio for now(default values are used in the paper, so they probably do not perform too terribly)
online_view_transform = torchvision.transforms.RandomResizedCrop(size=VIEW_SIZE)
target_view_transform = torchvision.transforms.RandomResizedCrop(size=VIEW_SIZE)

# unsimplified BYOL loss
def BYOL_loss(online_out, target_out):
    return ((target_out - online_out)**2) 

online_model = BYOL_model()
target_model = BYOL_model()
target_model.load_state_dict(online_model.state_dict()) # Copy parameters so the models match

TAU = 0.9995

# EMA update to target from online model
def update_target():
   with torch.no_grad():
      for online_param, target_param in zip(online_model.parameters(), target_model.parameters()):
         target_param.copy_(TAU * target_param + (1 - TAU)*online_param)

optimizer = torch.optim.AdamW(online_model.parameters(), lr = 3e-5)
optimizer.zero_grad(set_to_none=True)

# single weight update pass
def train_iteration(x):
    optimizer.zero_grad(set_to_none=True)
    online_model.train().to(device)
    target_model.train().to(device) #.train() should be unnecessary, but include it just incase

    x = x.to(device)
    online_x = online_view_transform(x)
    target_x = target_view_transform(x)

    online_z = online_model(online_x)
    with torch.no_grad():
      target_z = target_model(target_x).detach()

    loss = BYOL_loss(online_z, target_z).mean()
    loss.backward()
    optimizer.step()

    update_target()

    return loss.item()

# 1 epoch of training
def train_epoch(curr_epoch):
    loss_list = []
    pbar = tqdm(train_loader)
    for input, _ in pbar:
      curr_loss = train_iteration(input)
      loss_list.append(curr_loss)

      pbar.set_description(f"Epoch {curr_epoch}, Train Loss: {curr_loss}")

    return loss_list

# get average loss and outputs of val set
def test():
    online_model.eval().to(device)
    target_model.eval().to(device)
    out_list = []
    loss_list = []
    with torch.no_grad():
       pbar = tqdm(test_loader)
       for x, _ in pbar:
          x = x.to(device)

          online_z = online_model(x) # predictor q isn't necessary? <- argued in other papers
          target_z = target_model(x)

          loss_list.append(BYOL_loss(online_z, target_z))
          out_list.append(online_z)
    
    curr_test_loss = torch.cat(loss_list).mean().item()
    print("Test Loss: " + str(curr_test_loss))

    return curr_test_loss, torch.cat(out_list)

# main function
def train_and_test(num_epochs):
    train_loss_list = []
    test_loss_list = []
    for i in tqdm(range(num_epochs)):
      train_loss_list += train_epoch(i + 1)
      curr_test_loss, test_out = test()
      test_loss_list.append(curr_test_loss)

    return train_loss_list, test_loss_list, test_out

NUM_EPOCHS = 10

train_loss_list, test_loss_list, test_out = train_and_test(NUM_EPOCHS)

plt.plot(train_loss_list)
plt.savefig("train_loss.png")
plt.clf()

plt.plot(test_loss_list)
plt.savefig("test_loss.png")
plt.clf()

pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_out.cpu())
print(test_pca.shape)
plt.scatter(test_pca[:, 0], test_pca[:, 1])
plt.savefig("PCA.png")