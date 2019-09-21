import torch as th
import torchvision as tv
import PRO_GAN as pg
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
if __name__ == '__main__':
    
    device = th.device("cuda")
    
    # some parameters:
    depth = 8
    # hyper-parameters per depth (resolution)
    '''
    num_epochs = [10, 20, 20, 20,20,20,20,20,20]
    fade_ins = [50, 50, 50, 50,50,50,50,50,50]
    batch_sizes = [128, 128, 128, 128,64,32,16,8]
    latent_size = 512
    '''
    size = pow(2,depth+1)
    num_epochs = [10, 20, 20, 20,30,50,80,120]
    fade_ins = [50, 50, 50, 50,50, 50, 50, 50]
    batch_sizes = [128, 128, 128, 128,64,32,16,8]
    latent_size = 512
    root = "/data"
    # get the data. Ignore the test data and their classes
    dataset = ImageFolder(root = root,transform=transforms.Compose( [transforms.Resize(size=size,interpolation=Image.NEAREST),transforms.ToTensor()]))

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ConditionalProGAN(num_classes=1, depth=depth, 
                                    latent_size=latent_size,learning_rate=0.0007, device=device)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    pro_gan.train(
        dataset=root,
        size=size,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes
        
    )
    # ======================================================================  
