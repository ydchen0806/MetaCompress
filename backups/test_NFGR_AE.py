
import argparse
import sys

from omegaconf import OmegaConf
from utils.Networks import init_gmod, init_hy, init_phi
from utils.misc import configure_optimizer
from utils.CompressFramework import _BaseCompressFramerwork, eval_performance
from utils.Typing import CompressFrameworkOpt, NormalizeOpt,CropOpt, ReproducOpt, SingleTaskOpt, TransformOpt
from tqdm import tqdm
import math
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
from torch.cuda.amp import GradScaler,autocast
import random
from einops import rearrange,repeat
from utils.io import *
from utils.transform import *
from utils.dataset import create_flattened_coords
from utils.Logger import MyLogger
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
class FlattenedSampler:
    def __init__(self,data: torch.Tensor,sample_size:int,sample_count:int,) -> None:
        self.sample_size = sample_size
        self.sample_count = sample_count
        if len(data.shape) == 5:
            batch_size,data_channel,d,h,w = data.shape
            self.coords = create_flattened_coords((d,h,w))
            self.coords = repeat(self.coords,'pop c -> n pop c',n=batch_size)
            self.data = rearrange(data,'n data_channel d h w -> n (d h w) data_channel')
            self.pop_size = d*h*w
        else:
            raise NotImplementedError
    def __len__(self):
        return self.sample_count
    def __iter__(self,):
        self.index = 0
        return self
    def __next__(self,) -> Tuple[torch.Tensor,torch.Tensor]:
        if self.index < self.__len__():
            sampled_idxs = torch.randint(0,self.pop_size,(self.sample_size,))
            sampled_coords  = self.coords[:,sampled_idxs,:]
            sampled_data  = self.data[:,sampled_idxs,:]
            self.index += 1
            return sampled_coords,sampled_data
        else:
            raise StopIteration
class FlattenedDataset:
    def __init__(self,batch_size:int,Normalize_opt:NormalizeOpt,shuffle_path:bool=True,data_dir:str=None,data_path_list:List[str]=None) -> None:
        if (data_dir is not None) and (data_path_list is not None):
            raise "Only one args can be used !"
        if data_dir is not None:
            self.data_path_list = gen_pathlist_fromimgdir(data_dir)
        elif data_path_list is not None:
            self.data_path_list = data_path_list
        else:
            raise "At least one args is given !"
        self.batch_size = batch_size
        self.Normalize_opt = Normalize_opt
        self.shuffle_path = shuffle_path
    def __len__(self):
        return math.ceil(len(self.data_path_list)/self.batch_size)
    def __iter__(self,):
        if self.shuffle_path:
            random.shuffle(self.data_path_list)
        self.index = 0
        return self
    def __next__(self) -> Tuple[torch.Tensor,dict]: 
        if self.index < self.__len__():
            start_idx = self.index*self.batch_size
            end_idx = min(start_idx+self.batch_size,len(self.data_path_list))
            path_idxs = torch.arange(start_idx,end_idx)
            path_list = [self.data_path_list[idx] for idx in path_idxs]
            data = read_data_batch(path_list)
            data,sideinfos = normalize_data(data,**self.Normalize_opt)
            self.index += 1
            return data, sideinfos | {'data_shape':list(data.shape)}
        else:
            raise StopIteration

class NFGR_AE(_BaseCompressFramerwork):
    def __init__(self, opt: CompressFrameworkOpt,) -> None:
        super().__init__(opt)
        self.data_channel = self.opt.Module.phi.data_channel
        self.y_channel = self.opt.Module.gmod.y_channel
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
        self.module['gmod']  = init_gmod(dim_hidden=self.opt.Module.phi.features,layers=self.opt.Module.phi.layers-1,**self.opt.Module.gmod)
        self.module['hy'] = init_hy(y_channel=self.y_channel,data_channel=self.data_channel,**self.opt.Module.hy)
    def sample_nf(self, coords: torch.Tensor,mods:List[torch.Tensor]) -> torch.Tensor:
        data_hat = self.module['phi'].forward_syn_wocrop(coords,mods)
        return data_hat
    def loss_Distortion_func(self, coords: torch.Tensor,y:torch.Tensor,data_gt: torch.Tensor) -> torch.Tensor:
        mods = self.module['gmod'](y)
        data_hat = self.module['phi'].forward_syn_wocrop(coords,mods)
        loss_Distortion = F.mse_loss(data_hat,data_gt)
        return loss_Distortion
    def train(self,save_dir:str,Log:MyLogger):
        os.makedirs(opj(save_dir,'trained_module'),exist_ok=True)
        # device
        device = 'cuda' if self.opt.Train.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # dataset
        val_data_path_list_list = gen_data_path_list_list(self.opt.Train.val_data_dir,self.opt.Train.val_data_quanity)
        dataset = FlattenedDataset(self.opt.Train.batch_size,self.opt.Normalize,True,data_dir=self.opt.Train.train_data_dir)
        # optimizer_module
        optimizer_module = configure_optimizer(self.module_parameters(),self.opt.Train.optimizer_name_module,self.opt.Train.lr_module)
        # gradient descent
        max_steps = self.opt.Train.max_steps
        log_every_n_step = self.opt.Train.log_every_n_step
        val_every_n_step = self.opt.Train.val_every_n_step 
        val_every_n_epoch = self.opt.Train.val_every_n_epoch
        def start():
            pbar = tqdm(total=max_steps,desc='Training',file=sys.stdout)
            steps = 0
            scaler = GradScaler()
            for epoch in range(int(1e8)):
                for (data,sideinfos) in dataset:
                    data = data.to(device)
                    flattened_sampler = FlattenedSampler(data,self.opt.Train.sample_size,self.opt.Train.sample_count)
                    pbar.set_description_str("Training Epoch={}({} steps per epoch = {} dataset_iters* {} cropped_sampler_iters)".format(
                        epoch,dataset.__len__()*flattened_sampler.__len__(),dataset.__len__(),flattened_sampler.__len__()))   
                    for (sampled_coords,sampled_data) in flattened_sampler:
                        optimizer_module.zero_grad()
                        sampled_coords = sampled_coords.to(device)
                        # sampled_data = sampled_data.to(device)
                        y = self.module['hy'](data)
                        with autocast():
                            loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer_module)
                        scaler.update()
                        pbar.update(1)
                        steps += 1
                        if steps % log_every_n_step == 0:
                            Log.log_metrics({'remaining steps/train':max_steps - steps},steps)
                            Log.log_metrics({'loss/train':loss.item()},steps)
                            pbar.set_postfix_str('loss/train={:.6f}'.format(loss.item()))
                        if steps % val_every_n_step == 0:
                            # evaluate performance
                            self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                            eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                            performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                            Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                        'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                        if steps == max_steps:
                            self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                            eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                            performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                            Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                                'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                            return            
                if (epoch+1)%val_every_n_epoch==0:
                    # evaluate performance
                    self.save_module(opj(save_dir,'trained_module','epoch_{}_step_{}.pt'.format(epoch,steps)))
                    eval_save_dir = opj(save_dir,'eval_results','epoch_{}_step_{}'.format(epoch,steps))
                    performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                    Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                        'performance/compress_ratio':performance['compress_ratio'].mean(),'epoch':epoch},steps)
                    
        start()
        self.move_module_to('cpu')
        torch.cuda.empty_cache()
    def compress(self, data_path_list: List[str], save_path: str=None):
        # device
        device = 'cuda' if self.opt.Compress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        self.set_module_eval()
        self.set_module_no_grad()
        # dataset
        dataset = FlattenedDataset(len(data_path_list),self.opt.Normalize,True,data_path_list=data_path_list)
        data,sideinfos = iter(dataset).__next__()
        data = data.to(device)
        flattened_sampler = FlattenedSampler(data,self.opt.Compress.sample_size,self.opt.Compress.max_steps)
        # compressing
        y = self.module['hy'](data)
        y.requires_grad=True
        optimizer_y = configure_optimizer([{'params':y}],self.opt.Compress.optimizer_name_y,self.opt.Compress.lr_y)
        pbar = tqdm(total=self.opt.Compress.max_steps,desc='Compressing',leave=False,file=sys.stdout)
        scaler = GradScaler()
        for idx,(sampled_coords,sampled_data) in enumerate(flattened_sampler):
            sampled_coords = sampled_coords.to(device)
            with autocast():
                loss = self.loss_Distortion_func(sampled_coords,y,sampled_data)
            scaler.scale(loss).backward()
            scaler.step(optimizer_y)
            scaler.update()
            if (idx+1)%100 == 0:
                pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            pbar.update(1)
        compressed_data = {'sideinfos':sideinfos,'y':y.data.cpu()}
        if save_path is not None:
            torch.save(compressed_data,save_path)
        self.set_module_train()
        self.set_module_grad()
        return compressed_data
    def decompress(self, compressed_data_path: str=None, compressed_data:Dict[str,Union[torch.Tensor,Dict]]=None,save_path_list: List[str]=None,half:bool=True):
        # device
        device = 'cuda' if self.opt.Decompress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # decompress
        if compressed_data_path is not None:
            compressed_data = torch.load(compressed_data_path)
        sideinfos = compressed_data['sideinfos']
        y = compressed_data['y'].to(device)
        data_shape = sideinfos['data_shape']
        # calc mods Avoiding repetitive computation 
        mods = self.module['gmod'](y)
        # sample from nf
        batch_size,data_channel,*coords_shape = data_shape
        sample_size = self.opt.Decompress.sample_size
        with torch.no_grad():
            if len(coords_shape) == 3:
                d,h,w = coords_shape
                pop_size = d*h*w
                coords = create_flattened_coords((d,h,w)).to(device)
                coords = repeat(coords,'pop c -> n pop c',n=batch_size)
                flattened_data = torch.zeros((batch_size,pop_size,data_channel),device=device)
            else:
                raise NotImplementedError
            for index in tqdm(range(math.ceil(pop_size/sample_size)),'Decompressing',leave=False,file=sys.stdout):
                start_idx = index*sample_size
                end_idx = min(start_idx+sample_size,pop_size)
                sampled_coords = coords[:,start_idx:end_idx,:]
                flattened_data[:,start_idx:end_idx,:] = self.sample_nf(sampled_coords,mods)
            data = rearrange(flattened_data,'n (d h w) c -> n c d h w',n=batch_size,d=d,h=h,w=w)   
            data = data.cpu()
        data = invnormalize_data(data,sideinfos,**self.opt.Normalize)
        if save_path_list is not None:
            save_data_batch(data,save_path_list)
        return data

def reproduc(opt:ReproducOpt):
    """Make experiments reproducible
    """
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = opt.benchmark
    torch.backends.cudnn.deterministic = opt.deterministic
def main():
    reproduc(opt.Reproduc)
    Log = MyLogger(**opt.Log)
    Log.log_opt(opt)
    CompressFramework = NFGR_AE(opt.CompressFramework)
    CompressFramework.train(Log.logdir,Log)
    # eval_data_path_list_list = gen_data_path_list_list('D:\\Dataset\\xray\\prasad_patchesrandomresizecrop_500_500_200',1)
    # eval_performance(eval_data_path_list_list,CompressFramework,Log.logdir)
if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    parser = argparse.ArgumentParser(description='test_NFGR_AE')
    parser.add_argument('-p',type=str,default=opj(opd(__file__),'opt','SingleTask','NFGR_AE','default.yaml'),help='yaml file path')
    parser.add_argument('-g',type=str,default='0',help='which gpu to use')
    args = parser.parse_args()
    opt: SingleTaskOpt = OmegaConf.load(args.p)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.g)
    main()