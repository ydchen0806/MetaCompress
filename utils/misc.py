import sys
from matplotlib import pyplot as plt
import omegaconf.listconfig
import pandas as pd
from tqdm import tqdm
import math
import os
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from einops import rearrange,repeat
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from utils.dataset import create_flattened_coords
from utils.io import *
from utils.ssim import ssim as ssim_calc
from utils.ssim import ms_ssim as ms_ssim_calc
from copy import deepcopy
from scipy import ndimage
def omegaconf2list(opt,prefix='',sep = '.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix,k,v))
            # if k in ['iter_list','step_list']: # do not sparse list
            #     dot_notation_list.append("{}{}={}".format(prefix,k,v))
            # else:
            #     templist = []
            #     for v_ in v:
            #         templist.append('{}{}={}'.format(prefix,k,v_))
            #     dot_notation_list.append(templist)   
        elif isinstance(v,(float,str,int,)):
            notation_list.append("{}{}={}".format(prefix,k,v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix,k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep,sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list
def omegaconf2dotlist(opt,prefix='',):
    return omegaconf2list(opt,prefix,sep='.')
def omegaconf2dict(opt,sep):
    notation_list = omegaconf2list(opt,sep=sep)
    dict = {notation.split('=')[0]:notation.split('=')[1] for notation in notation_list}
    return dict
def reconstruct_flattened(data_shape:tuple,sample_size:int,sample_nf:Callable,device:str='cpu',half:bool=False,coords_mode:str='-1,1') -> torch.Tensor:
    """用于decompress
    """
    batch_size,data_channel,*coords_shape = data_shape
    assert batch_size == 1
    # sample
    with torch.no_grad():
        if len(coords_shape) == 2:
            h,w = coords_shape
            pop_size = h*w
            coords = create_flattened_coords((h,w),coords_mode).to(device)
            flattened_data = torch.zeros((pop_size,data_channel),device=device)
            if half:
                flattened_data = flattened_data.half()
        elif len(coords_shape) == 3:
            d,h,w = coords_shape
            pop_size = d*h*w
            coords = create_flattened_coords((d,h,w),coords_mode).to(device)
            flattened_data = torch.zeros((pop_size,data_channel),device=device)
            if half:
                flattened_data = flattened_data.half()
        else:
            raise NotImplementedError
        for index in tqdm(range(math.ceil(pop_size/sample_size)),'Decompressing',leave=False,file=sys.stdout):
            start_idx = index*sample_size
            end_idx = min(start_idx+sample_size,pop_size)
            sampled_coords = coords[start_idx:end_idx,:]
            if half:
                sampled_coords = sampled_coords.half()
            flattened_data[start_idx:end_idx,:] = sample_nf(sampled_coords)
        if len(coords_shape) == 2:
            data = rearrange(flattened_data,'(n h w) c -> n c h w',n=1,h=h,w=w)   
        elif len(coords_shape) == 3:
            data = rearrange(flattened_data,'(n d h w) c -> n c d h w',n=1,d=d,h=h,w=w)   
    return data
def reconstruct_cropped(data_shape:tuple,sample_size:int,mods:List[torch.Tensor],sample_nf:Callable,ps_h:int,ps_w:int,ol_h:int,ol_w:int,ps_d:int=None,ol_d:int=None,device:str='cpu') -> torch.Tensor:
    """用于decompress
    """
    batch_size,data_channel,*coords_shape = data_shape
    # sample
    with torch.no_grad():
        if len(coords_shape) == 2:
            return NotImplementedError
        elif len(coords_shape) == 3:
            d,h,w = coords_shape
            pc_d = math.ceil((d-ol_d)/(ps_d-ol_d))
            pc_h = math.ceil((h-ol_h)/(ps_h-ol_h))
            pc_w = math.ceil((w-ol_w)/(ps_w-ol_w))
            pop_size = ps_d*ps_h*ps_w
            coords = create_flattened_coords((ps_d,ps_h,ps_w)).to(device)
            coords = repeat(coords,'pop c -> n pc_d pc_h pc_w pop c',n=batch_size,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
            cropped_data = torch.zeros((batch_size,pc_d,pc_h,pc_w,pop_size,data_channel),device=device)
        else:
            raise NotImplementedError
        for index in tqdm(range(math.ceil(pop_size/sample_size)),'Decompressing',leave=False,file=sys.stdout):
            start_idx = index*sample_size
            end_idx = min(start_idx+sample_size,pop_size)
            sampled_coords = coords[...,start_idx:end_idx,:]
            cropped_data[...,start_idx:end_idx,:] = sample_nf(sampled_coords,mods)
        cropped_data = cropped_data.cpu()
    # merge
    if len(coords_shape) == 2:
        return NotImplementedError
    elif len(coords_shape) == 3:
        cropped_data = rearrange(cropped_data,'n pc_d pc_h pc_w (ps_d ps_h ps_w) c -> n pc_d pc_h pc_w c ps_d ps_h ps_w',n = batch_size,ps_d = ps_d,ps_h = ps_h,ps_w = ps_w)
        # merge 
        data = torch.zeros((batch_size,data_channel,*coords_shape))
        weights = torch.zeros((batch_size,data_channel,*coords_shape))
        #FIXME 复现论文中的线性插值加权方法
        weights_patch = torch.zeros((batch_size,data_channel,ps_d,ps_h,ps_w))
        center_idx = (ps_d//2,ps_h//2,ps_w//2)
        for d_idx in range(pc_d):
            for h_idx in range(pc_h):
                for w_idx in range(pc_w):
                    weights_patch[...,d_idx,h_idx,w_idx] = math.sqrt((d_idx-center_idx[0])**2+(h_idx-center_idx[1])**2+(w_idx-center_idx[2])**2)
        weights_patch = torch.abs(weights_patch-weights_patch.max())+1
        for pc_d_idx in range(pc_d):
            if pc_d_idx == 0:
                data_d_start = 0
            elif pc_d_idx == pc_d-1:
                data_d_start = d-ps_d
            else:
                data_d_start = pc_d_idx*(ps_d-ol_d)
            for pc_h_idx in range(pc_h):
                if pc_h_idx == 0:
                    data_h_start = 0
                elif pc_h_idx == pc_h-1:
                    data_h_start = h-ps_h
                else:
                    data_h_start = pc_h_idx*(ps_h-ol_h)
                for pc_w_idx in range(pc_w):
                    if pc_w_idx == 0:
                        data_w_start = 0
                    elif pc_w_idx == pc_w-1:
                        data_w_start = w-ps_w
                    else:
                        data_w_start = pc_w_idx*(ps_w-ol_w)
                    data[...,data_d_start:data_d_start+ps_d,data_h_start:data_h_start+ps_h,data_w_start:data_w_start+ps_w] += cropped_data[:,pc_d_idx,pc_h_idx,pc_w_idx,...]*weights_patch
                    weights[...,data_d_start:data_d_start+ps_d,data_h_start:data_h_start+ps_h,data_w_start:data_w_start+ps_w] += weights_patch
        data = data/weights
    else:
        raise NotImplementedError
    return data
def loss_bpp_func(likelihoods:torch.Tensor) -> torch.Tensor:
    """bits per pixel
    """
    if len(likelihoods.shape) == 5:
        n,c,d,h,w = likelihoods.shape
        num_pixels = d*h*w*n
    elif len(likelihoods.shape) == 4:
        n,c,h,w = likelihoods.shape
        num_pixels = h*w*n
    else:
        raise NotImplementedError
    loss_bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    return loss_bpp
def configure_optimizer(parameters,optimizer:str,lr:float) -> torch.optim.Optimizer:
    if optimizer == 'Adam':
        Optimizer = torch.optim.Adam(parameters,lr=lr)
    elif optimizer == 'Adamax':
        Optimizer = torch.optim.Adamax(parameters,lr=lr)
    elif optimizer == 'SGD':
        Optimizer = torch.optim.SGD(parameters,lr=lr)
    else:
        raise NotImplementedError
    return Optimizer
def configure_lr_scheduler(optimizer,lr_scheduler_opt):
    lr_scheduler_opt = deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop('name')
    if lr_scheduler_name == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'none':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000000000])
    else:
        raise NotImplementedError
    return lr_scheduler
def gradient_descent(loss:torch.Tensor,optimizer_list:List[torch.optim.Optimizer]):
    for optimizer in optimizer_list:
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizer_list:
        optimizer.step()
def init_y(batch_size:int,y_channel:int,pc_shape:tuple,device:str='cpu') -> torch.Tensor:
    y = nn.Parameter(torch.empty((batch_size,y_channel,*pc_shape),device=device))
    nn.init.xavier_normal_(y,10000)
    return y
def init_z(batch_size:int,z_channel:int,pc_shape:tuple,device:str='cpu') -> torch.Tensor:
    z = nn.Parameter(torch.empty((batch_size,z_channel,*pc_shape),device=device))
    nn.init.xavier_normal_(z,10000)
    return z
def annealed_temperature(t:int, r:float, ub:float, lb:float=1e-8, scheme:str='exp', t0:int=700):
    """Return the temperature at time step t, based on a chosen annealing schedule.
    Args:
        t (int): step/iteration number
        r (float): decay strength
        ub (float): maximum/init temperature
        lb (float, optional): small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
        scheme (str, optional): [description]. Defaults to 'exp'.
        t0 (int, optional): [description]. Defaults to 700.fixes temperature at ub for initial t0 iterations
    """
    if scheme == 'exp':
        tau = math.exp(-r * t)
    elif scheme == 'exp0':
        # Modified version of above that fixes temperature at ub for initial t0 iterations
        tau = ub * math.exp(-r * (t - t0))
    elif scheme == 'linear':
        # Cool temperature linearly from ub after the initial t0 iterations
        tau = -r * (t - t0) + ub
    else:
        raise NotImplementedError
    return min(max(tau, lb), ub)
def mip_ops(data:np.ndarray,save_dir:Union[None,str]=None,data_name:str='',suffix:str='') -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert len(data.shape) == 5
    mip_d = data.max(2)
    mip_h = data.max(3)
    mip_w = data.max(4)
    if save_dir is not None:
        save_data_batch(mip_d,[opj(save_dir,data_name+'_mip_d'+suffix)])
        save_data_batch(mip_h,[opj(save_dir,data_name+'_mip_h'+suffix)])
        save_data_batch(mip_w,[opj(save_dir,data_name+'_mip_w'+suffix)])
    return mip_d,mip_h,mip_w
def preprocess(data:np.ndarray,denoise_level:int,denoise_close:Union[bool,List[int]],clip_range:List[int]):
    if denoise_close == False:
        data[data<=denoise_level]= 0
    else:
        data[ndimage.binary_opening(data<=denoise_level, structure=np.ones(tuple([1,1]+list(denoise_close))),iterations=1)]=0
    data = data.clip(*clip_range)
    return data
def parse_checkpoints(checkpoints:Union[str,int],max_steps:int):
    if checkpoints == 'none':
        checkpoints = [max_steps]
    elif 'every' in checkpoints:
        _,interval = checkpoints.split('_')
        interval = int(interval)
        checkpoints = list(range(interval,max_steps,interval))
        checkpoints.append(max_steps)
    elif isinstance(checkpoints,int):
        if checkpoints >= max_steps:
            checkpoints = [max_steps]
        else:
            checkpoints = [checkpoints,max_steps]
    else:
        checkpoints = [int(s) for s in checkpoints.split(",") if int(s) < max_steps]
        checkpoints.append(max_steps)
    return checkpoints
def parse_weight(data:np.ndarray,weight_type_list:List[str]):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    weight = np.ones_like(data).astype(np.float32)
    for weight_type in weight_type_list:
        if 'quantile' in weight_type:
            _,ge_thres,ql,qh,scale = weight_type.split('_')
            ge_thres,ql,qh,scale = float(ge_thres),float(ql),float(qh),float(scale)
            l = np.quantile(data[data>=ge_thres],ql)
            h = np.quantile(data[data>=ge_thres],qh)
            weight[(data>=l) * (data<=h)] = scale
        elif 'value' in weight_type:
            _,l,h,scale = weight_type.split('_')
            l,h,scale = float(l),float(h),float(scale)
            weight[(data>=l) * (data<=h)] = scale
        elif weight_type == 'none':
            pass
        else:
            raise NotImplementedError
    return weight
def plot_conv3d_weight(name:str,weight:torch.Tensor,savedir:str):
    weight = weight.cpu().numpy()
    for fig_idx in range(weight.shape[0]):
        fig = plt.figure(figsize = (20,20))
        figname = '{}_out_channel_{}'.format(name,fig_idx)
        fig.suptitle(figname)
        for row_idx in range(weight.shape[1]):
            for col_idx in range(weight.shape[2]):
                weight_ = weight[fig_idx,row_idx,col_idx]
                fig.add_subplot(weight.shape[1],weight.shape[2],row_idx*weight.shape[2]+col_idx+1)
                im = fig.axes[-1].imshow(weight_,cmap='Greys_r',vmin=weight_.min(),vmax=weight_.max())
                fig.axes[-1].set_xticks([])
                fig.axes[-1].set_yticks([])
                fig.axes[-1].set_title('in_channel:{} d:{}'.format(row_idx,col_idx))
                for i in range(weight.shape[3]):
                    for j in range(weight.shape[4]):
                        text = fig.axes[-1].text(j, i,'{:.5f}'.format(weight_[j,i]),size=5,
                                    ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig(opj(savedir,figname+'.png'))
    pass
def divide_data(data:np.ndarray,divide_type:str,):
    assert len(data.shape) == 5
    if 'every' in divide_type:
        data_chunk_list = []
        _,chunk_d,chunk_h,chunk_w = divide_type.split('_')
        chunk_d,chunk_h,chunk_w = int(chunk_d),int(chunk_h),int(chunk_w)
        dsections = [i for i in range(data.shape[2]) if i%chunk_d==0]
        hsections = [i for i in range(data.shape[3]) if i%chunk_h==0]
        wsections = [i for i in range(data.shape[4]) if i%chunk_w==0]
        dsections.append(data.shape[2])
        hsections.append(data.shape[3])
        wsections.append(data.shape[4])
        for di in range(len(dsections)-1):
            for hi in range(len(hsections)-1):
                for wi in range(len(wsections)-1):
                    data_chunk_list.append({'data':data[:,:,dsections[di]:dsections[di+1],hsections[hi]:hsections[hi+1],wsections[wi]:wsections[wi+1]],'d':[dsections[di],dsections[di+1]-1],'h':[hsections[hi],hsections[hi+1]-1],'w':[wsections[wi],wsections[wi+1]-1]})
    else:
        raise NotImplementedError
    for data_chunk in data_chunk_list:
        data_chunk['total_size'] = data.size
        data_chunk['size'] = data_chunk['data'].size
        data_chunk['name'] = 'd_{}_{}-h_{}_{}-w_{}_{}'.format(*data_chunk['d'],*data_chunk['h'],*data_chunk['w'])
    return data_chunk_list
def alloc_param(data_chunk_list:List[dict],param_size:float,param_alloc:str,param_size_thres:float):
    if param_alloc == 'by_size':
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*(data_chunk['size'])/data_chunk['total_size']
    elif param_alloc == 'by_aoi':
        aoi_total = 0
        for data_chunk in data_chunk_list:
            aoi_total += (data_chunk['data']>0).sum()  # 已经preprocess 所以用0做阈值
        for data_chunk in data_chunk_list:
            data_chunk['param_size'] = param_size*(data_chunk['data']>0).sum()/aoi_total
            data_chunk['param_size'] = float(data_chunk['param_size'])
    # filter the too small param_size
    data_chunk_list_ = [data_chunk for data_chunk in data_chunk_list if data_chunk['param_size'] >= param_size_thres]
    if len(data_chunk_list_) < len(data_chunk_list):
        return alloc_param(data_chunk_list_,param_size,param_alloc,param_size_thres)
    # for data_chunk in data_chunk_list_:
    #     data_chunk['name'] += '-param_size_{}'.format(int(data_chunk['param_size']))
    return data_chunk_list_
def merge_divided_data(decompressed_data_chunk_list:List[dict],data_shape):
    decompressed_data_dtype = decompressed_data_chunk_list[0]['data'].dtype
    decompressed_data = np.zeros(data_shape,dtype=np.float32)
    for decompressed_data_chunk in decompressed_data_chunk_list:
        dstart,dend = decompressed_data_chunk['d']
        hstart,hend = decompressed_data_chunk['h']
        wstart,wend = decompressed_data_chunk['w']
        decompressed_data[:,:,dstart:dend+1,hstart:hend+1,wstart:wend+1] += decompressed_data_chunk['data']
    if decompressed_data_dtype == np.uint16:
        max = 65535
    elif decompressed_data_dtype == np.uint8:
        max = 255
    decompressed_data = decompressed_data.clip(None,max)
    decompressed_data = decompressed_data.astype(decompressed_data_dtype)
    return decompressed_data
def cal_psnr(data_gt:np.ndarray,data_hat:np.ndarray,data_range,psnr_type:dict):
    weight = parse_weight(data_gt,psnr_type['weight'])
    data_hat = np.copy(data_hat)
    data_gt = np.copy(data_gt)
    if 'clip' in psnr_type.keys():
        data_gt = data_gt.clip(psnr_type['clip'][0],psnr_type['clip'][1])
        data_hat = data_hat.clip(psnr_type['clip'][0],psnr_type['clip'][1])
    mse = np.mean(np.power(data_gt/data_range-data_hat/data_range,2)*weight)
    psnr = -10*np.log10(mse)
    return psnr
def cal_aoi(data_gt:np.ndarray,data_hat:np.ndarray,thres:float):
    hat = np.copy(data_hat)
    gt = np.copy(data_gt)
    hat[data_hat>=thres]=1
    hat[data_hat<thres]=0
    gt[data_gt>=thres]=1
    gt[data_gt<thres]=0
    aoi = 1-np.mean(np.abs(gt-hat))
    return aoi
def eval_performance(steps:int,orig_data:np.ndarray,decompressed_data:np.ndarray,Log,aoi_thres_list:List[int],psnr_type_list:List[str],ssim:bool):
    # performance
    dtype = orig_data.dtype.name
    orig_data = orig_data.astype(np.float32)
    decompressed_data = decompressed_data.astype(np.float32)
    if dtype == 'uint8':
        max = 255
    elif dtype == 'uint12':
        max = 4098
    elif dtype == 'uint16':
        max = 65535
    else:
        raise NotImplementedError
    for aoi_thres in aoi_thres_list:
        aoi_value = cal_aoi(orig_data,decompressed_data,aoi_thres)
        Log.log_metrics({'aoi-{}'.format(aoi_thres):aoi_value},steps)
    for psnr_type in psnr_type_list:
        psnr_value = cal_psnr(orig_data,decompressed_data,max,psnr_type)
        Log.log_metrics({'psnr-{}'.format(str(psnr_type)):psnr_value},steps)
    if ssim:
        orig_data = torch.from_numpy(orig_data)
        decompressed_data = torch.from_numpy(decompressed_data)
        if len(orig_data.shape) == 4:
            ssim_value = ssim_calc(orig_data,decompressed_data,max)
        elif len(orig_data.shape) == 5:
            if orig_data.shape[2] <= 8:
                ssim_value = ssim_calc(orig_data,decompressed_data,max)
            else:
                try:
                    ssim_value = ms_ssim_calc(orig_data,decompressed_data,max)
                except:
                    ssim_value = ssim_calc(orig_data,decompressed_data,max)
        else:
            raise ValueError
        Log.log_metrics({'ssim':ssim_value},steps)