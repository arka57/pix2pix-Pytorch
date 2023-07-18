import torch
from torchvision.utils import save_image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_some_examples(gen, val_loader, epoch, folder):
    c=0
    for index,(x,y) in  enumerate(val_loader):
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/y_gen_{c}_{epoch}.png")
            save_image(x * 0.5 + 0.5, folder + f"/input_{c}_{epoch}.png")
            if epoch == 1:
                save_image(y * 0.5 + 0.5, folder + f"/label_{c}_{epoch}.png")
        c=c+1        
        gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=train.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr