import argparse
import sys
from omegaconf import OmegaConf
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
from collections import OrderedDict
from utils.torchmeta import get_subdict,MetaModule, MetaSequential
import struct
from utils.MetaNet import *
import copy
import argparse
from torch.utils.tensorboard import SummaryWriter


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
            # return sampled_coords,sampled_data
            return self.coords, self.data
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

class FlattenedDataset_RAM:
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
        data = read_data_batch(self.data_path_list)
        self.data,self.dtype = normalize_data(data,**self.Normalize_opt)
    def __len__(self):
        return math.ceil(len(self.data_path_list)/self.batch_size)
    def __iter__(self,):
        if self.shuffle_path:
            self.path_idxs = torch.randperm(len(self.data_path_list))
        self.index = 0
        return self
    def __next__(self) -> Tuple[torch.Tensor,dict]: 
        if self.index < self.__len__():
            start_idx = self.index*self.batch_size
            end_idx = min(start_idx+self.batch_size,len(self.data_path_list))
            path_idxs = self.path_idxs[start_idx:end_idx]
            data = self.data[path_idxs]
            self.index += 1
            return data,{'dtype':self.dtype,'data_shape':list(data.shape)}
        else:
            raise StopIteration

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class MetaFC(MetaModule):
    '''A fully connected neural network that allows swapping out the weights, either via a hypernetwork
    or via MAML.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features),
            nn.ReLU(inplace=True)
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features),
                nn.ReLU(inplace=True)
            ))

        if outermost_linear:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features),
            ))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features),
                nn.ReLU(inplace=True)
            ))

        self.net = MetaSequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, coords, params=None, **kwargs):
        '''Simple forward pass without computation of spatial gradients.'''
        output = self.net(coords, params=get_subdict(params, 'net'))
        return output


class SineLayer(MetaModule):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = float(omega_0)

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, params=None):
        intermed = self.linear(input, params=get_subdict(params, 'linear'))
        return torch.sin(self.omega_0 * intermed)


class Siren(MetaModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., special_first=True):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        layer = SineLayer

        self.net = []
        self.net.append(layer(in_features, hidden_features,
                              is_first=special_first, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(layer(hidden_features, hidden_features,
                                  is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30.,
                                             np.sqrt(6 / hidden_features) / 30.)
            self.net.append(final_linear)
        else:
            self.net.append(layer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, params=None):
        x = coords

        for i, layer in enumerate(self.net):
            x = layer(x, params=get_subdict(params, f'net.{i}'))

        return x
    
    @staticmethod
    def calc_features(param_count, coords_channel, data_channel, layers, **kwargs):
        a = layers - 2
        b = coords_channel + 1 + layers - 2 + data_channel
        c = -param_count + data_channel

        if a == 0:
            features = round(-c / b)
        else:
            features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features
    
    
def init_weights_normal(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


class MAML(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static': 
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter': # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([])
            for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module(query_x, params=fast_params)
        return output

    def generate_params(self, sampled_coords,sampled_data):
        """Specializes the model"""
        x = sampled_coords
        y = sampled_data

        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the 
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = self.hypo_module(x, params=fast_params)

                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions)

                if float(loss.item()) > prev_loss:
                    print('inner lr too high?')
                
                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, sampled_coords,sampled_data, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(sampled_coords,sampled_data)

        # Compute the final outputs.
        model_output = self.hypo_module(sampled_coords, params=fast_params)
        out_dict = {'model_out':model_output, 'intermed_predictions':intermed_predictions}

        return out_dict



def l2_loss(prediction, gt):
    return F.mse_loss(prediction, gt)
    

def save_model_(model,save_path,devive:str='cuda'):
    if hasattr(model, "net"):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        for l in range(len(model.net)):
            if l < len(model.net) - 1:
                weight = model.net[l].linear.weight.data.to('cpu')
                bias = model.net[l].linear.bias.data.to('cpu')
            else:
                weight = model.net[l].weight.data.to('cpu')
                bias = model.net[l].bias.data.to('cpu')
            # weight = copy.deepcopy(weight).to('cpu')
            # bias = copy.deepcopy(bias).to('cpu')
            weight_save_path = os.path.join(save_path,'weight-{}-{}-{}'.format(l,weight.shape[0],weight.shape[1]))
            weight = np.array(weight).reshape(-1)
            with open(weight_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(weight), *weight))
            bias_save_path = os.path.join(save_path,'bias-{}-{}'.format(l,len(bias)))
            with open(bias_save_path, 'wb') as data_file:
                data_file.write(struct.pack('f'*len(bias), *bias))
            if l < len(model.net) - 1:
                weight = model.net[l].linear.weight.data.to(devive)
                bias = model.net[l].linear.bias.data.to(devive)
            else:
                weight = model.net[l].weight.data.to(devive)
                bias = model.net[l].bias.data.to(devive)
    else:
        model = torch.save(model.state_dict(), save_path)


def reproduc(opt:ReproducOpt):
    """Make experiments reproducible
    """
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = opt.benchmark
    torch.backends.cudnn.deterministic = opt.deterministic


def MAML_train(args):
    
    reproduc(opt.Reproduc)

    maml_folder = opt.Save_dir
    os.makedirs(maml_folder, exist_ok=True)
    tblogger = SummaryWriter(maml_folder)
    OmegaConf.save(opt, opj(maml_folder, "config.yaml"))
    
    # device
    device = 'cuda' if opt.Train.gpu else 'cpu'

    # prepare dataset
    dataset = FlattenedDataset(opt.Train.batch_size, opt.Normalize, True, data_dir=opt.Train.train_data_dir)
    val_dataset = FlattenedDataset(opt.Train.batch_size, opt.Normalize, True, data_dir=opt.Train.val_data_dir)

    # define the model
    data_temp = gen_data_path_list_list(opt.Train.val_data_dir, 1)
    ideal_network_size_bytes = os.path.getsize(data_temp[0][0]) / opt.Siren.compression_ratio
    ideal_network_parameters_count = ideal_network_size_bytes / 4.0
    n_network_features = Siren.calc_features(
        ideal_network_parameters_count, opt.Siren.coords_channel,
        opt.Siren.data_channel, opt.Siren.layers)
    model = Siren(
        opt.Siren.coords_channel, n_network_features, opt.Siren.layers-2,
        opt.Siren.data_channel, True, opt.Siren.w0)
    model = model.to(device)
    if args.use_method == 'MAML':
        meta_phi = MAML(num_meta_steps=opt.Train.max_inner_steps, hypo_module=model,
                        loss=l2_loss, init_lr=opt.Train.lr_phi_inner,lr_type=opt.Train.lr_type)
    elif args.use_method == 'Reptile':
        meta_phi = Reptile(num_meta_steps=opt.Train.max_inner_steps, hypo_module=model,
                           loss=l2_loss, init_lr=opt.Train.lr_phi_inner, lr_type=opt.Train.lr_type)
    elif args.use_method == 'ANIL':
        meta_phi = ANIL(num_inner_steps=opt.Train.max_inner_steps, hypo_module=model,
                        loss=l2_loss, init_lr=opt.Train.lr_phi_inner, lr_type=opt.Train.lr_type)
    elif args.use_method == 'MetaSGD':
        meta_phi = MetaSGD(num_meta_steps=opt.Train.max_inner_steps, hypo_module=model,
                           loss=l2_loss, init_lr=opt.Train.lr_phi_inner, lr_type=opt.Train.lr_type)
    elif args.use_method == 'MAMLWithSSL':
        meta_phi = MAMLWithSSL(num_meta_steps=opt.Train.max_inner_steps, hypo_module=model,
                               task_loss=l2_loss, ssl_loss=l2_loss, init_lr=opt.Train.lr_phi_inner,
                               lr_type=opt.Train.lr_type)
    else:
        raise NotImplementedError
    
    meta_phi = meta_phi.to(device)
    optim = torch.optim.Adam(lr=opt.Train.lr_phi_outer, params=meta_phi.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, eps=1e-08,
                                                           verbose=True)
    best_val_loss = float("Inf")
    best_state_dict = copy.deepcopy(meta_phi.state_dict())
    steps = opt.Train.max_inner_steps
    step = 0

    for i in range(opt.Train.maml_epoch):
        for (data,sideinfos) in dataset:
            step += 1
            data = data.to(device)
            flattened_sampler = FlattenedSampler(data, opt.Train.sample_size, opt.Train.sample_count)
            for (sampled_coords,sampled_data) in flattened_sampler:
                sampled_coords = sampled_coords.to(device)
                model_output = meta_phi(sampled_coords, sampled_data)
                loss = ((model_output['model_out'] - sampled_data) ** 2).mean()

                if not step % opt.Train.summary_every_n_step:
                    tblogger.add_scalar('meta trian/train loss', loss.item(), step)
                    lr_cur = optim.state_dict()['param_groups'][0]['lr']
                    tblogger.add_scalar(f"meta trian/lr", lr_cur, step)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if not step % opt.Train.val_every_n_step:
                    val_loss_sum = 0
                    meta_phi.first_order = True
                    with torch.no_grad():
                        for val_step, (val_data, sideinfos) in enumerate(val_dataset):
                            val_data = val_data.to(device)
                            val_flattened_sampler = FlattenedSampler(val_data, opt.Train.sample_size, opt.Train.sample_count)
                            for (val_sampled_coords,val_sampled_data) in val_flattened_sampler:
                                val_sampled_coords = val_sampled_coords.to(device)
                                meta_phi.float()
                                val_model_output = meta_phi(val_sampled_coords, val_sampled_data)
                                val_loss = ((val_model_output['model_out'] - val_sampled_data) ** 2).mean().detach().cpu()
                                val_loss_sum += val_loss
                        print("validation loss: ", val_loss_sum.item())
                        tblogger.add_scalar('meta trian/validation loss', val_loss_sum.item() / len(val_dataset), step)
                        if val_loss_sum < best_val_loss:
                            best_state_dict = copy.deepcopy(meta_phi.state_dict())
                            best_val_loss = val_loss_sum
                        scheduler.step(val_loss_sum)
                    meta_phi.first_order = False        
    meta_phi.load_state_dict(best_state_dict)
    model = meta_phi.hypo_module
    torch.save(model.state_dict(),
        os.path.join(os.path.join(maml_folder, 'model_maml.pth')))
    torch.save(meta_phi.state_dict(),
                os.path.join(os.path.join(maml_folder, 'maml_obj.pth')))
    tblogger.close()


if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='MAML training')
    parser.add_argument('-p',type=str,default=opj(opd(__file__),'Run_pipeline','config_w0_5.yaml'),help='yaml file path')
    parser.add_argument('-g',type=str,default='0',help='which gpu to use')
    parser.add_argument('--use_method', type=str, default='MAML', help='MAML, Reptile, ANIL, MetaSGD, MAMLWithSSL')
    args = parser.parse_args()
    opt: SingleTaskOpt = OmegaConf.load(args.p)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.g)

    MAML_train(args)