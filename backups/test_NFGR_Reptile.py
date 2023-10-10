import argparse
import sys
from omegaconf import OmegaConf
from utils.Networks import init_phi
from utils.misc import configure_optimizer, reconstruct_flattened
from utils.CompressFramework import _BaseCompressFramerwork, eval_performance
from utils.Typing import CompressFrameworkOpt, NormalizeOpt,CropOpt, ReproducOpt, SingleTaskOpt, TransformOpt
from tqdm import tqdm
import math
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
import torch.nn as nn
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
from copy import deepcopy
from torch.multiprocessing import set_start_method
import operator
from collections import OrderedDict
from numbers import Number
try:
    set_start_method('spawn')
except Exception as e:
    print(e)

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
    def __init__(self,outer_steps:int,outer_batch_size:int,Normalize_opt:NormalizeOpt,data_dir:str=None,data_path_list:List[str]=None) -> None:
        if (data_dir is not None) and (data_path_list is not None):
            raise "Only one args can be used !"
        if data_dir is not None:
            self.data_path_list = gen_pathlist_fromimgdir(data_dir)
        elif data_path_list is not None:
            self.data_path_list = data_path_list
        else:
            raise "At least one args is given !"
        self.outer_steps = outer_steps
        self.outer_batch_size = outer_batch_size
        self.Normalize_opt = Normalize_opt
        self.path_len = len(self.data_path_list)
    def __len__(self):
        return self.outer_steps
    def __iter__(self,):
        self.index = 0
        return self
    def __next__(self) -> Tuple[torch.Tensor,dict]: 
        if self.index < self.__len__():
            path_idxs = torch.randint(0,self.path_len,(self.outer_batch_size,))
            path_list = [self.data_path_list[idx] for idx in path_idxs]
            data = read_data_batch(path_list)
            data,sideinfos = normalize_data(data,**self.Normalize_opt)
            self.index += 1
            return data, sideinfos | {'data_shape':list(data.shape)}
        else:
            raise StopIteration

class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

def inner_sgd(phi_opt,weights_init,inner_lr_phi,data,inner_sample_size,inner_steps,device,weights_optimized_dict,inner_loss_log_dict,idx):
    phi = init_phi(**phi_opt).half()
    phi.load_state_dict(weights_init)
    optimizer_phi_inner = torch.optim.SGD(phi.parameters(),lr=inner_lr_phi)
    data = data.to(device)
    flattened_sampler = FlattenedSampler(data,inner_sample_size,inner_steps)
    phi = phi.to(device)
    for (sampled_coords,sampled_data) in flattened_sampler:
        sampled_coords = sampled_coords.to(device)
        loss = F.mse_loss(phi(sampled_coords.half()),sampled_data.half())
        loss.backward()
        phi.float()
        optimizer_phi_inner.step()
        phi.half()
    weights_optimized = phi.cpu().state_dict()
    weights_optimized_dict[str(idx)] = weights_optimized
    inner_loss_log_dict[str(idx)] = loss.item()
class NeuralFiledsGlobalRep_Reptile(_BaseCompressFramerwork):
    def __init__(self, opt: CompressFrameworkOpt,) -> None:
        super().__init__(opt)
        self.data_channel = self.opt.Module.phi.data_channel
        self.init_module()
    def init_module(self):
        self.module['phi'] = init_phi(**self.opt.Module.phi)
    def sample_nf(self, coords: torch.Tensor) -> torch.Tensor:
        data_hat = self.module['phi'].forward(coords)
        return data_hat
    def loss_Distortion_func(self, coords: torch.Tensor,data_gt: torch.Tensor) -> torch.Tensor:
        data_hat = self.module['phi'].forward(coords)
        loss_Distortion = F.mse_loss(data_hat,data_gt)
        return loss_Distortion
    def train(self,save_dir:str,Log:MyLogger):
        os.makedirs(opj(save_dir,'trained_module'),exist_ok=True)
        # device
        device = 'cuda' if self.opt.Train.gpu else 'cpu'
        # dataset
        outer_steps = self.opt.Train.outer_steps
        outer_batch_size = self.opt.Train.outer_batch_size
        val_data_path_list_list = gen_data_path_list_list(self.opt.Train.val_data_dir,self.opt.Train.val_data_quanity)
        dataset = FlattenedDataset(outer_steps,outer_batch_size,self.opt.Normalize,data_dir=self.opt.Train.train_data_dir)
        # optimizer_module
        self.module['phi'].half()
        # gradient descent
        log_every_n_step = self.opt.Train.log_every_n_step
        val_every_n_step = self.opt.Train.val_every_n_step 
        val_every_n_epoch = self.opt.Train.val_every_n_epoch
        inner_steps = self.opt.Train.inner_steps
        inner_sample_size = self.opt.Train.inner_sample_size
        inner_lr_phi = self.opt.Train.inner_lr_phi
        phi_opt = self.opt.Module.phi
        outerpbar = tqdm(dataset,desc='Training outer steps',file=sys.stdout)
        global weights_optimized_dict,inner_loss_log_dict
        for current_outer_steps,(data_batch,sideinfos) in enumerate(outerpbar):
            weights_init = self.module['phi'].cpu().state_dict()
            # inner optimize
            processes = []
            for idx in range(outer_batch_size):
                p = torch.multiprocessing.Process(target=inner_sgd, args=(phi_opt,weights_init,inner_lr_phi,data_batch[idx:idx+1],inner_sample_size,inner_steps,device,weights_optimized_dict,inner_loss_log_dict,idx))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            # outer optimize
            weights_optimized_list = [ParamDict(w) for w in weights_optimized_dict.values()]
            weights_optimized_sum = sum(weights_optimized_list, 0 * ParamDict(weights_init) )
            outer_lr_phi = self.opt.Train.outer_lr_phi * (1 - current_outer_steps / outer_steps) # linear schedule
            weights_init = ParamDict(weights_init)+(weights_optimized_sum-ParamDict(weights_init))*outer_lr_phi
            self.module['phi'].load_state_dict(weights_init)
            if current_outer_steps % log_every_n_step == 0:
                inner_final_loss = sum(inner_loss_log_dict.values())/outer_batch_size
                Log.log_metrics({'remaining steps/train':outer_steps - current_outer_steps},outer_steps)
                Log.log_metrics({'loss/train':inner_final_loss},outer_steps)
                outerpbar.set_postfix_str('loss/train={:.6f}'.format(inner_final_loss))
            if outer_steps % val_every_n_step == 0:# or steps == 1:
                # evaluate performance
                torch.cuda.empty_cache()
                self.save_module(opj(save_dir,'trained_module','current_outer_steps_{}.pt'.format(current_outer_steps)))
                eval_save_dir = opj(save_dir,'eval_results','current_outer_steps_{}'.format(current_outer_steps))
                performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
                Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
                    'performance/compress_ratio':performance['compress_ratio'].mean()},current_outer_steps)
        # final
        self.save_module(opj(save_dir,'trained_module','current_outer_steps_{}.pt'.format(current_outer_steps)))
        eval_save_dir = opj(save_dir,'eval_results','current_outer_steps_{}'.format(current_outer_steps))
        performance = eval_performance(val_data_path_list_list,self,eval_save_dir,True)
        Log.log_metrics({'performance/mse':performance['mse'].mean(),'performance/psnr':performance['psnr'].mean(),
            'performance/compress_ratio':performance['compress_ratio'].mean()},current_outer_steps)
        self.move_module_to('cpu')
        torch.cuda.empty_cache()
    def compress(self, data_path_list: List[str], save_path: str=None,):
        # device
        device = 'cuda' if self.opt.Compress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # dataset
        dataset = FlattenedDataset(1,len(data_path_list),self.opt.Normalize,data_path_list=data_path_list)
        data,sideinfos = iter(dataset).__next__()
        data = data.to(device)
        flattened_sampler = FlattenedSampler(data,self.opt.Compress.sample_size,self.opt.Compress.max_steps)
        # compressing
        optimizer_phi = configure_optimizer(self.module['phi'].parameters(),self.opt.Compress.optimizer_name_phi,self.opt.Compress.lr_phi)
        pbar = tqdm(flattened_sampler,desc='Compressing',leave=False,file=sys.stdout)
        for (sampled_coords,sampled_data) in pbar:
            optimizer_phi.zero_grad()
            sampled_coords = sampled_coords.to(device)
            sampled_data = sampled_data.to(device)
            self.module['phi'].half()
            data_hat = self.module['phi'].forward(sampled_coords.half())
            loss = F.mse_loss(data_hat,sampled_data.half())
            loss.backward()
            self.module['phi'].float()
            optimizer_phi.step()
            pbar.set_postfix_str("loss={:.6f}".format(loss.item()))

        self.module['phi'].half()
        compressed_data = {'sideinfos':sideinfos,'phi':self.module['phi'].cpu().state_dict()}
        if save_path is not None:
            torch.save(compressed_data,save_path)
        torch.cuda.empty_cache()
        return compressed_data
    def decompress(self, compressed_data_path: str=None, compressed_data:Dict[str,Union[torch.Tensor,Dict]]=None,save_path_list: List[str]=None):
        # device
        device = 'cuda' if self.opt.Decompress.gpu else 'cpu'
        # module
        self.move_module_to(device)
        # decompress
        if compressed_data_path is not None:
            compressed_data = torch.load(compressed_data_path)
        sideinfos = compressed_data['sideinfos']
        self.module['phi'].load_state_dict(compressed_data['phi'])
        self.move_module_to(device)
        data_shape = sideinfos['data_shape']
        # sample from nf
        data = reconstruct_flattened(data_shape,self.opt.Decompress.sample_size,self.sample_nf,device=device,half=True).float()
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
    CompressFramework = NeuralFiledsGlobalRep_Reptile(opt.CompressFramework)
    CompressFramework.train(Log.logdir,Log)
    # eval_data_path_list_list = gen_data_path_list_list('D:\\Dataset\\xray\\prasad_patchesrandomresizecrop_500_500_200',1)
    # eval_performance(eval_data_path_list_list,CompressFramework,Log.logdir)
if __name__=='__main__':
    # this manager can upate global var
    manager = torch.multiprocessing.Manager()
    weights_optimized_dict = manager.dict()
    inner_loss_log_dict = manager.dict()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    parser = argparse.ArgumentParser(description='test_NFGR_AE')
    parser.add_argument('-p',type=str,default=opj(opd(__file__),'opt','SingleTask','NFGR_Reptile','default.yaml'),help='yaml file path')
    parser.add_argument('-g',type=str,default='0',help='which gpu to use')
    args = parser.parse_args()
    opt: SingleTaskOpt = OmegaConf.load(args.p)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.g)
    main()