import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2






#Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"



def train_fn(dis,gen,loader,opt_dis, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop=tqdm(loader,leave=True)

    for index,(x,y) in enumerate(loop):
        x=x.to(DEVICE)
        y=y.to(DEVICE)


        #Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake=gen(x)
            D_real=dis(x,y)
            D_fake=dis(x,y_fake.detach())
            D_real_loss=bce(D_real,torch.ones_like(D_real))
            D_fake_loss=bce(D_fake,torch.zeros_like(D_fake))
            D_loss=(D_real_loss+D_fake_loss)/2

        opt_dis.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_dis)
        d_scaler.update()    


        #Train Generator
        with torch.cuda.amp.autocast():
            D_fake=dis(x,y_fake)
            G_fake_loss=bce(D_fake,torch.ones_like(D_fake))
            L1=l1_loss(y_fake,y)*L1_LAMBDA
            G_loss=G_fake_loss+L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()   


        if index % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )  




def main():
    dis = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_dis = optim.Adam(dis.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen,LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC, dis, opt_dis,LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train_fn(
            dis, gen, train_loader, opt_dis, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if SAVE_MODEL and epoch % 100 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(dis, opt_dis, filename=CHECKPOINT_DISC)

        if epoch%100==0:
            save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()