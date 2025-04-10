import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.autograd import Variable

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator, GeneratorM2CM_C
from model import UNet
from model import Discriminator 
from datasets import ImageDataset
import tqdm 
import numpy as np
import os
from pytorch_ssim import SSIM
import gc



saveDirType = 'MRI Colorization'
modelNameType = 'MRI Colorization'
load_model = False
downscale_input = False
load_checkpoint = 0
per_epoch_save = 10

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12335"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        netG_A2B: torch.nn.Module,
        netG_B2A: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        netC: torch.nn.Module,
        train_data: DataLoader,
        optimizer_G: torch.optim.Optimizer,
        optimizer_S: torch.optim.Optimizer,
        optimizer_D_A: torch.optim.Optimizer,
        optimizer_D_B: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.netG_A2B = netG_A2B.to(gpu_id)
        self.netG_B2A = netG_B2A.to(gpu_id)
        self.netD_A = netD_A.to(gpu_id)
        self.netD_B = netD_B.to(gpu_id)
        self.netC = netC.to(gpu_id)

        self.train_data = train_data
        self.optimizer_G = optimizer_G
        self.optimizer_S = optimizer_S
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.save_every = save_every
        self.model1 = DDP(netG_A2B, device_ids=[gpu_id])
        self.model2 = DDP(netG_B2A, device_ids=[gpu_id])
        self.model3 = DDP(netD_A, device_ids=[gpu_id])
        self.model4 = DDP(netD_B, device_ids=[gpu_id])
        self.model5 = DDP(netC, device_ids=[gpu_id])


    def _run_batch(self, real_A, real_B, real_C, target_real, target_fake, epoch):
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_ssim11 = SSIM()
        criterion_ssim9 = SSIM(window_size=9)
        criterion_ssim7 = SSIM(window_size=7)
        criterion_ssim5 = SSIM(window_size=5)
        criterion_ssim3 = SSIM(window_size=3)
        criterion_seg = torch.nn.CrossEntropyLoss()
        
        ###### Generators A2B and B2A ######
        self.optimizer_G.zero_grad()
        
        # GAN loss
        fake_B, genCryo = self.netG_A2B(real_A)
        pred_fake = self.netD_B(fake_B).view(-1)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        

        fake_A = self.netG_B2A(real_B)
        pred_fake = self.netD_A(fake_A).view(-1)
        
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        

        # Cycle loss
        recovered_A = self.netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B, _ = self.netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        rgb2grey = fake_B.clone().mean(dim=1).view(real_A.shape[0], 1, real_A.shape[2], real_A.shape[3])
        loss_multiscale = (1 - criterion_ssim11(rgb2grey, real_A)) + (1-criterion_ssim9(rgb2grey, real_A)) + (1-criterion_ssim7(rgb2grey, real_A)) + (1-criterion_ssim5(rgb2grey, real_A)) + (1-criterion_ssim3(rgb2grey, real_A) )
                        
        
        ## Segmentation Loss
        self.netC.eval()
        fake_C2 = self.netC(genCryo)
        real_C = real_C.to(torch.float32)
        loss_segmentation = criterion_seg(fake_C2, real_C)

        loss_ssim_cryo = (1 - criterion_ssim11(genCryo, real_B)) + (1-criterion_ssim9(genCryo, real_B)) + (1-criterion_ssim7(genCryo, real_B)) + (1-criterion_ssim5(genCryo, real_B)) + (1-criterion_ssim3(genCryo, real_B) )
        
        real_B_grey = real_B.clone().mean(dim=1).view(real_A.shape[0], 1, real_A.shape[2], real_A.shape[3])
        loss_ssim_cryo2mri = (1 - criterion_ssim11(fake_A, real_B_grey)) + (1-criterion_ssim9(fake_A, real_B_grey)) + (1-criterion_ssim7(fake_A, real_B_grey)) + (1-criterion_ssim5(fake_A, real_B_grey)) + (1-criterion_ssim3(fake_A, real_B_grey) )
        del real_B_grey
        gc.collect()
        
        # Total loss
        loss_G =  loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB  +  loss_multiscale+ loss_segmentation + loss_ssim_cryo#+ loss_identity_A + loss_identity_B
        loss_G.backward()
        self.optimizer_G.step()
        
        ###################################
        


        # Real loss
        pred_real = self.netD_A(real_A).view(-1)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = self.netD_A(fake_A.detach()).view(-1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        self.optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        self.optimizer_D_B.zero_grad()

        # Real loss
        pred_real = self.netD_B(real_B).view(-1)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        pred_fake = self.netD_B(fake_B.detach()).view(-1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        
        loss_D_B.backward()

        self.optimizer_D_B.step()
        
        epoch_checkpoint = load_checkpoint + epoch
         
        if epoch%per_epoch_save == 0:

            if not os.path.exists('./epoch_checkpoints/{}'.format(epoch_checkpoint)):
                os.mkdir('./epoch_checkpoints/{}'.format(epoch_checkpoint))

            torch.save(self.netG_A2B.state_dict(), 'epoch_checkpoints/{}/netG_A2B_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netG_B2A.state_dict(), 'epoch_checkpoints/{}/netG_B2A_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netD_A.state_dict(), 'epoch_checkpoints/{}/netD_A_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netD_B.state_dict(), 'epoch_checkpoints/{}/netD_B_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netC.state_dict(), 'epoch_checkpoints/{}/netC_{}.pth'.format(epoch_checkpoint, modelNameType))        




    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)))
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        Tensor = torch.cuda.FloatTensor 
        input_A = Tensor(b_sz, 1, 256, 256)
        input_B = Tensor(b_sz, 1, 256, 256)
        target_real = Variable(Tensor(b_sz).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(b_sz).fill_(0.0), requires_grad=False)
        downscale = torch.nn.Upsample(size=None, scale_factor=0.25, mode='bilinear').to(self.gpu_id)
        downscaleC = torch.nn.Upsample(size=None, scale_factor=0.25, mode='nearest').to(self.gpu_id)
        for source in self.train_data:

            sourceA = source['A'].to(self.gpu_id)
            sourceB = source['B'].to(self.gpu_id)
            sourceC = source['C'].to(self.gpu_id)

            if downscale_input:
                sourceA = downscale(sourceA)
                sourceB = downscale(sourceB)
                sourceC = downscaleC(sourceC)
                
            target_real = torch.ones(sourceA.shape[0]).to(torch.float32).to(self.gpu_id)
            target_fake = torch.zeros(sourceA.shape[0]).to(torch.float32).to(self.gpu_id)
            self._run_batch(sourceA, sourceB, sourceC, target_real, target_fake, epoch)


    
    def train(self, max_epochs: int):
        for epoch in tqdm.tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)



def load_train_objs():
    image_dir = 'path/to/your/image_dir'  # replace with your image directory
    label_dir = 'path/to/your/label_dir'  # a dictionary mapping image filenames to labels
    
    train_set = ImageDataset(image_dir, label_dir, transform=None) # load your dataset
    netG_A2B = GeneratorM2CM_C(1, 3)  
    netG_B2A = Generator(3, 1)  
    netD_A = Discriminator(1)  # load your model
    netD_B = Discriminator(3)
    netC = UNet(3, 47)

        
    params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())
    paramsC = list(netC.parameters())
    optimizer_G = torch.optim.Adam(params, lr=1e-4)
    optimizer_S = torch.optim.Adam(paramsC, lr=1e-4)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=1e-4)
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=1e-4)
    return train_set, netG_A2B, netG_B2A, netD_A, netD_B, netC, optimizer_G, optimizer_S, optimizer_D_A, optimizer_D_B  

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset,  netG_A2B, netG_B2A, netD_A, netD_B, netC, optimizer_G, optimizer_S, optimizer_D_A, optimizer_D_B  = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(netG_A2B, netG_B2A, netD_A, netD_B, netC, train_data, optimizer_G, optimizer_S, optimizer_D_A, optimizer_D_B, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    save_data_dir = 'path/to/your/save_data_dir'  # replace with your save data directory

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=28, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    

    if not os.path.exists('./output/generated{}'.format(saveDirType)):
            os.mkdir('./output/generated{}'.format(saveDirType))

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)