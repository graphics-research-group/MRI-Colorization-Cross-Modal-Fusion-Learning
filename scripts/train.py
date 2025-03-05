import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from datautils import MyTrainDataset2

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
from model import Discriminator ####################################
from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
import wandb
import tqdm 
import skimage
import numpy as np
import os
from pytorch_ssim import SSIM
import gc
import tqdm



saveDirType = 'revised_pipeline_end2end'
modelNameType = 'revised_pipeline_end2end'
load_model = False
downscale_input = False
load_checkpoint = 0
WANDB = False
colormap = {i:np.random.randint(0, 255, (3,)) for i in range(1, 47)}
colormap[0] = np.array([0, 0, 0])
if WANDB:
    wandb.init(project='revised_pipeline_end2end')

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
        for i in range(2):
            ###### Generators A2B and B2A ######
            self.optimizer_G.zero_grad()
            
            # GAN loss
            fake_B, genCryo = self.netG_A2B(real_A)
            pred_fake = self.netD_B(fake_B).view(-1)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            #### new loss hsv
            # hsvConverted_fake_B = get_hsv(fake_B)
            # hsvConverted_real_B = get_hsv(real_B)

            # # loss_hsv = criterion_identity(rgb2hsv_torch(fake_B.clone().detach()).to(torch.float32), rgb2hsv_torch(real_B).to(torch.float32)).requires_grad_(True)
            # loss_hsv = criterion_identity(hsvConverted_fake_B, hsvConverted_real_B)

            fake_A = self.netG_B2A(real_B)
            pred_fake = self.netD_A(fake_A).view(-1)
            
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            # loss_ssimB2A = criterion_ssim(fake_A, real_A)
            

            # Cycle loss
            recovered_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B, _ = self.netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
            rgb2grey = fake_B.clone().mean(dim=1).view(real_A.shape[0], 1, real_A.shape[2], real_A.shape[3])
            loss_multiscale = (1 - criterion_ssim11(rgb2grey, real_A)) + (1-criterion_ssim9(rgb2grey, real_A)) + (1-criterion_ssim7(rgb2grey, real_A)) + (1-criterion_ssim5(rgb2grey, real_A)) + (1-criterion_ssim3(rgb2grey, real_A) )
                            
            del rgb2grey
            gc.collect()
            ## Segmentation Loss
            self.netC.eval()
            fake_C2 = self.netC(genCryo)
            real_C = real_C.to(torch.float32)
            loss_segmentation2 = criterion_seg(fake_C2, real_C)

            loss_ssim_cryo = (1 - criterion_ssim11(genCryo, real_B)) + (1-criterion_ssim9(genCryo, real_B)) + (1-criterion_ssim7(genCryo, real_B)) + (1-criterion_ssim5(genCryo, real_B)) + (1-criterion_ssim3(genCryo, real_B) )
            
            real_B_grey = real_B.clone().mean(dim=1).view(real_A.shape[0], 1, real_A.shape[2], real_A.shape[3])
            loss_ssim_cryo2mri = (1 - criterion_ssim11(fake_A, real_B_grey)) + (1-criterion_ssim9(fake_A, real_B_grey)) + (1-criterion_ssim7(fake_A, real_B_grey)) + (1-criterion_ssim5(fake_A, real_B_grey)) + (1-criterion_ssim3(fake_A, real_B_grey) )
            del real_B_grey
            gc.collect()
            # Total loss
            loss_G =  loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB  +  loss_multiscale+ loss_segmentation2 + loss_ssim_cryo#+ loss_identity_A + loss_identity_B
            loss_G.backward()
            self.optimizer_G.step()
        
        ###################################
        ### Segnet
        self.netC.train()
        self.optimizer_S.zero_grad()
        fake_C = self.netC(real_B)
        real_C = real_C.to(torch.float32)
        loss_segmentation = criterion_seg(fake_C, real_C)
        loss_segmentation.backward()
        self.optimizer_S.step()
        ###### Discriminator A ######
        self.optimizer_D_A.zero_grad()


        # Real loss
        pred_real = self.netD_A(real_A).view(-1)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        # fake_A = fake_A_buffer.push_and_pop(fake_A)
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
        # fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = self.netD_B(fake_B.detach()).view(-1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        
        loss_D_B.backward()

        self.optimizer_D_B.step()

        real_A = ((real_A.permute(0, 2, 3, 1).cpu().detach().numpy())*255).astype(np.uint8)
        real_B = ((real_B.permute(0, 2, 3, 1).cpu().detach().numpy())*255).astype(np.uint8)
        fake_B = ((fake_B.permute(0, 2, 3, 1).cpu().detach().numpy())*255).astype(np.uint8)
        fake_A = ((fake_A.permute(0, 2, 3, 1).cpu().detach().numpy())*255).astype(np.uint8)
        genCryo = ((genCryo.permute(0, 2, 3, 1).cpu().detach().numpy())*255).astype(np.uint8)
        
        # real_C =  torch.argmax(real_C, dim=1).cpu().detach().numpy()
        # fake_C = torch.argmax(fake_C, dim=1).cpu().detach().numpy()
        # print(real_C.shape, fake_C.shape)
        if WANDB:
            wandb.log({'lossG {}'.format('cycle2D') : loss_G.mean().item()})
            wandb.log({'lossDB {}'.format('cycle2D') : loss_D_B.mean().item()})
            wandb.log({'lossDA {}'.format('cycle2D') : loss_D_A.mean().item()})
            wandb.log({'loss_GAN_A2B' : loss_GAN_A2B.mean().item()})
            wandb.log({'loss_GAN_B2A' : loss_GAN_B2A.mean().item()})
            wandb.log({'loss_cycle_ABA': loss_cycle_ABA.mean().item()})
            wandb.log({'loss_cycle_BAB': loss_cycle_BAB.mean().item()})
            wandb.log({'loss_cycle_BAB': loss_cycle_BAB.mean().item()})
            wandb.log({'loss seg ': loss_segmentation.mean().item()})
            wandb.log({'loss ms_ssim mri': loss_multiscale.mean().item()})
            wandb.log({'loss ms_ssim cryo': loss_ssim_cryo.mean().item()})
            wandb.log({'loss ms_ssim cryo': loss_ssim_cryo2mri.mean().item()})
            wandb.log({'loss seg ': loss_segmentation2.mean().item()})

        epoch_checkpoint = load_checkpoint + epoch
        if epoch_checkpoint%5==0:
            skimage.io.imsave('./output/generated{}/{}_mri_trainset_1.png'.format(saveDirType, epoch_checkpoint), real_A[0, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_cryo_trainset_1.png'.format(saveDirType, epoch_checkpoint), real_B[0, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_trainset_1.png'.format(saveDirType, epoch_checkpoint), fake_B[0, ...])
            skimage.io.imsave('./output/generated{}/{}_genMRI_trainset_1.png'.format(saveDirType, epoch_checkpoint), fake_A[0, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_seg_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(real_C[0, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegB_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C[0, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_segLabel_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_labelmap(real_C[0, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegLabelB_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C[0, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genMRI2Cryo_1.png'.format(saveDirType, epoch_checkpoint), genCryo[0, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_SegLabelB_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C2[0, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryoSegB_trainset_1.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C2[0, ...], colormap))


            skimage.io.imsave('./output/generated{}/{}_genMRI2Cryo_2.png'.format(saveDirType, epoch_checkpoint), genCryo[1, ...])
            skimage.io.imsave('./output/generated{}/{}_mri_trainset_2.png'.format(saveDirType, epoch_checkpoint), real_A[1, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_cryo_trainset_2.png'.format(saveDirType, epoch_checkpoint), real_B[1, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_trainset_2.png'.format(saveDirType, epoch_checkpoint), fake_B[1, ...])
            skimage.io.imsave('./output/generated{}/{}_genMRI_trainset_2.png'.format(saveDirType, epoch_checkpoint), fake_A[1, :256, :256, 0]) 
            skimage.io.imsave('./output/generated{}/{}_seg_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(real_C[1, ...], colormap)) 
            skimage.io.imsave('./output/generated{}/{}_genSegB_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C[1, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_segLabel_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_labelmap(real_C[1, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegLabelB_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C[1, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryo_SegLabelB_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C2[1, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryoSegB_trainset_2.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C2[1, ...], colormap))


            skimage.io.imsave('./output/generated{}/{}_genMRI2Cryo_3.png'.format(saveDirType, epoch_checkpoint), genCryo[2, ...])
            skimage.io.imsave('./output/generated{}/{}_mri_trainset_3.png'.format(saveDirType, epoch_checkpoint), real_A[2, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_cryo_trainset_3.png'.format(saveDirType, epoch_checkpoint), real_B[2, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_trainset_3.png'.format(saveDirType, epoch_checkpoint), fake_B[2, ...])
            skimage.io.imsave('./output/generated{}/{}_genMRI_trainset_3.png'.format(saveDirType, epoch_checkpoint), fake_A[2, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_seg_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(real_C[2, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegB_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C[2, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_segLabel_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_labelmap(real_C[2, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegLabelB_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C[2, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryo_SegLabelB_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C2[2, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryoSegB_trainset_3.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C2[2, ...], colormap))

            
            skimage.io.imsave('./output/generated{}/{}_genMRI2Cryo_4.png'.format(saveDirType, epoch_checkpoint), genCryo[3, ...])
            skimage.io.imsave('./output/generated{}/{}_mri_trainset_4.png'.format(saveDirType, epoch_checkpoint), real_A[3, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_cryo_trainset_4.png'.format(saveDirType, epoch_checkpoint), real_B[3, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_trainset_4.png'.format(saveDirType, epoch_checkpoint), fake_B[3, ...])
            skimage.io.imsave('./output/generated{}/{}_genMRI_trainset_4.png'.format(saveDirType, epoch_checkpoint), fake_A[3, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_seg_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(real_C[3, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegB_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C[3, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_segLabel_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_labelmap(real_C[3, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegLabelB_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C[3, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryo_SegLabelB_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C2[3, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryoSegB_trainset_4.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C2[3, ...], colormap))

            
            skimage.io.imsave('./output/generated{}/{}_genMRI2Cryo_5.png'.format(saveDirType, epoch_checkpoint), genCryo[4, ...])
            skimage.io.imsave('./output/generated{}/{}_mri_trainset_5.png'.format(saveDirType, epoch_checkpoint), real_A[4, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_cryo_trainset_5.png'.format(saveDirType, epoch_checkpoint), real_B[4, ...])
            skimage.io.imsave('./output/generated{}/{}_genCryo_trainset_5.png'.format(saveDirType, epoch_checkpoint), fake_B[4, ...])
            skimage.io.imsave('./output/generated{}/{}_genMRI_trainset_5.png'.format(saveDirType, epoch_checkpoint), fake_A[4, :256, :256, 0])
            skimage.io.imsave('./output/generated{}/{}_seg_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(real_C[4, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegB_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C[4, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_segLabel_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_labelmap(real_C[4, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genSegLabelB_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C[4, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryo_SegLabelB_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_labelmap(fake_C2[4, ...], colormap))
            skimage.io.imsave('./output/generated{}/{}_genCryoSegB_trainset_5.png'.format(saveDirType, epoch_checkpoint), get_coloredSeg(fake_C2[4, ...], colormap))

         
        if epoch%10 == 0:

            if not os.path.exists('./epoch_checkpoints/{}'.format(epoch_checkpoint)):
                os.mkdir('./epoch_checkpoints/{}'.format(epoch_checkpoint))

            torch.save(self.netG_A2B.state_dict(), 'epoch_checkpoints/{}/netG_A2B_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netG_B2A.state_dict(), 'epoch_checkpoints/{}/netG_B2A_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netD_A.state_dict(), 'epoch_checkpoints/{}/netD_A_{}.pth'.format(epoch_checkpoint, modelNameType))
            torch.save(self.netD_B.state_dict(), 'epoch_checkpoints/{}/netD_B_{}.pth'.format(epoch_checkpoint, modelNameType))
            # torch.save(netG_A2A.state_dict(), 'epoch_checkpoints/{}/netG_A2A_{}.pth'.format(epoch, modelNameType))
            # torch.save(netG_B2B.state_dict(), 'epoch_checkpoints/{}/netG_B2B_{}.pth'.format(epoch, modelNameType))
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
                
            # target_real = torch.ones(sourceA.shape[0]*256*256).to(torch.float32).to(self.gpu_id)
            # target_fake = torch.zeros(sourceA.shape[0]*256*256).to(torch.float32).to(self.gpu_id)
            target_real = torch.ones(sourceA.shape[0]).to(torch.float32).to(self.gpu_id)
            target_fake = torch.zeros(sourceA.shape[0]).to(torch.float32).to(self.gpu_id)
            # targets = targets.to(self.gpu_id)
            self._run_batch(sourceA, sourceB, sourceC, target_real, target_fake, epoch)

    def _save_checkpoint(self, epoch):
        # ckp1 = self.netG_A2B.module.state_dict()
        PATH = "checkpoint1.pt"
        # torch.save(ckp1, PATH)
        torch.save(self.netG_A2B.state_dict(), 'epoch_checkpoints/netG_A2B_{}.pth'.format(modelNameType))
        torch.save(self.netG_B2A.state_dict(), 'epoch_checkpoints/netG_B2A_{}.pth'.format(modelNameType))
        torch.save(self.netD_A.state_dict(), 'epoch_checkpoints/netD_A_{}.pth'.format(modelNameType))
        torch.save(self.netD_B.state_dict(), 'epoch_checkpoints/netD_B_{}.pth'.format(modelNameType))
        torch.save(self.netC.state_dict(), 'epoch_checkpoints/net_C{}.pth'.format(modelNameType))

        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        # ckp2 = self.netG_B2A.module.state_dict()
        # PATH = "checkpoint2.pt"
        # torch.save(ckp2, PATH)
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        # ckp3 = self.netD_A.module.state_dict()
        # PATH = "checkpoint3.pt"
        # torch.save(ckp3, PATH)
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        # ckp4 = self.netD_B.module.state_dict()
        # PATH = "checkpoint4.pt"
        # torch.save(ckp4, PATH)
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        # ckp5 = self.netC.module.state_dict()
        # PATH = "checkpoint5.pt"
        # torch.save(ckp5, PATH)
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def train(self, max_epochs: int):
        for epoch in tqdm.tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def get_files(filename):
    filenames_all = open(filename, 'r')
    files_ = filenames_all.readlines()

    files =[]
    for i in range(len(files_)):
       files.append(files_[i].split('\n')[0])

    return files

def get_coloredSeg(channelwise_bitmap, colormap, n_classes=47):
    segmap = torch.argmax(channelwise_bitmap, dim=0)
    segmap_np = segmap.cpu().detach().numpy()
    outColored = np.zeros((segmap_np.shape[0], segmap_np.shape[1], 3)).astype(np.uint8)
    for i in range(n_classes):
        outColored[np.where(segmap_np==i)] = colormap[i]
    return outColored.astype(np.uint8)
def get_labelmap(channelwise_bitmap, colormap, n_classes=47):
    segmap = torch.argmax(channelwise_bitmap, dim=0)
    segmap_np = segmap.cpu().detach().numpy()
    outColored = np.zeros((segmap_np.shape[0], segmap_np.shape[1], 3)).astype(np.uint8)
    for i in range(n_classes):
        outColored[np.where(segmap_np==i)] = i
    return outColored.astype(np.uint8)
def load_train_objs():
    axia_data = get_files('./axisDataL2.txt')
    trainsetFiles = get_files('./train_filesNoBlacks256L2.txt') #######################
    trainsetFiles2 = get_files('./train_filesNoBlacks256L2_2.txt')
    trainsetFiles_ = axia_data + trainsetFiles
    trainset_filesPath = [('/home/ojaswa/mayuri/Projects/data/data2D_256/trainL2', trainsetFiles_[i]) for i in range(len(trainsetFiles_))]
    trainset_filesPath2 = [('/home/ojaswa/mayuri/Projects/data/Slices256L2_test', trainsetFiles2[i]) for i in range(len(trainsetFiles2))]
    save_data_dir = '/home/ojaswa/mayuri/Projects/data/data2D_256/'
    trainsetFiles = trainset_filesPath + trainset_filesPath2
    train_set = ImageDataset6_5_noerrors(trainsetFiles, transform=transforms.ToTensor(), mode='trainL2', n_channels=55) # load your dataset
    netG_A2B = GeneratorM2CM_C(1, 3)  # load your model
    netG_B2A = Generator(3, 1)  # load your model
    netD_A = Discriminator(1)  # load your model
    netD_B = Discriminator(3)
    netC = UNet(3, 47)

    if load_model:
        # netG_A2B.load_state_dict(torch.load('./output/netG_A2B_{}.pth'.format('multiscale_l2_multigpu_down')))
        # netG_B2A.load_state_dict(torch.load('./output/netG_B2A_{}.pth'.format('multiscale_l2_multigpu_down')))
        netC.load_state_dict(torch.load('./epoch_checkpoints/160/netC{}.pth'.format('_multiscale_l2_multigpu_256_pretrained'), map_location='cuda:0'))
        # netD_A.load_state_dict(torch.load('./output/netD_A_{}.pth'.format('multiscale_l2_multigpu_down')))
        # netD_B.load_state_dict(torch.load('./output/netD_B_{}.pth'.format('multiscale_l2_multigpu_down')))
        print("Models loaded successfully")

    params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())# + list(netC.parameters())
    paramsC = list(netC.parameters())
    optimizer_G = torch.optim.Adam(params, lr=1e-3)
    optimizer_S = torch.optim.Adam(paramsC, lr=1e-4)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=1e-6)
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=1e-6)
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
    # axia_data = get_files('./axisData.txt')
    # trainsetFiles = get_files('./train_filesNoBlacks256.txt') #######################
    # trainsetFiles = axia_data + trainsetFiles


    save_data_dir = '/home/ojaswa/mayuri/Projects/data/data2D_256/'

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=28, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    

    if not os.path.exists('./output/generated{}'.format(saveDirType)):
            os.mkdir('./output/generated{}'.format(saveDirType))

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)