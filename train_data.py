import argparse
from omegaconf import OmegaConf
from utils.misc import configure_optimizer, reconstruct_flattened
from utils.Typing import CompressFrameworkOpt, NormalizeOpt,CropOpt, ReproducOpt, SingleTaskOpt, TransformOpt
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
from utils.io import *
from utils.transform import *
from utils.dataset import create_flattened_coords
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from collections import OrderedDict
import copy
import math

from torch.utils.tensorboard import SummaryWriter
from MAML import FlattenedSampler, FlattenedDataset, Siren, MAML, l2_loss, reproduc, save_model_
import warnings
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING, ReduceLROnPlateau
import tifffile
from skimage.metrics import structural_similarity
import csv


EXPERIMENTAL_CONDITIONS = ["data_name", "data_type", "data_shape", "actual_ratio"]
METRICS = [
    "epochs",
    "psnr",
    "ssim",
    "mse",
    "original_data_path",
    "decompressed_data_path",
]
EXPERIMENTAL_RESULTS_KEYS = (
    ["algorithm_name"] + EXPERIMENTAL_CONDITIONS + METRICS + ["config_path"]
)


def get_type_max(data):
    dtype = data.dtype.name
    if dtype == "uint8":
        max = 255
    elif dtype == "uint12":
        max = 4098
    elif dtype == "uint16":
        max = 65535
    elif dtype == "float32":
        max = 65535
    elif dtype == "float64":
        max = 65535
    else:
        raise NotImplementedError
    return max


def calc_mse_psnr(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    mse = np.mean(np.power(predicted / data_range - gt / data_range, 2))
    psnr = -10 * np.log10(mse)
    return mse, psnr


def calc_ssim(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    ssim = structural_similarity(gt, predicted, data_range=data_range)
    return ssim


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    # Reference from "https://github.com/YannickStruempler/inr_based_compression"
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, warmup_end_lr=0, warmup_steps=0, verbose=False):

        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold,
                         threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
        self.warmup_end_lr = warmup_end_lr
        self.warmup_steps = warmup_steps
        self._set_warmup_lr(1)

    def _set_warmup_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):

            new_lr = epoch * (self.warmup_end_lr / self.warmup_steps)
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: increase learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    # Override step method to include warmup
    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.warmup_steps > 0 and epoch <= self.warmup_steps:
            self._set_warmup_lr(epoch)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def refine(maml, context_dict, steps):
    """Specializes the model"""
    x = context_dict.get('x').to(device)
    y = context_dict.get('y').to(device)

    meta_batch_size = x.shape[0]

    with torch.enable_grad():
        # First, replicate the initialization for each batch item.
        # This is the learned initialization, i.e., in the outer loop,
        # the gradients are backpropagated all the way into the
        # "meta_named_parameters" of the hypo_module.
        fast_params = OrderedDict()
        for name, param in maml.hypo_module.meta_named_parameters():
            fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

        prev_loss = 1e6
        intermed_predictions = []
        for j in range(steps):
            # Using the current set of parameters, perform a forward pass with the context inputs.
            predictions = maml.hypo_module(x, params=fast_params)

            # Compute the loss on the context labels.
            loss = maml.loss(predictions, y)
            intermed_predictions.append(predictions.detach().cpu())

            if loss > prev_loss:
                print('inner lr too high?')

            fast_params, grads = maml._update_step(loss, fast_params, j)

            prev_loss = loss

    return fast_params, intermed_predictions


def train_from_meta():
    reproduc(opt.Reproduc)

    maml_folder = opt.Save_dir
    overfit_folder = opj(maml_folder, 'overfit')
    os.makedirs(overfit_folder, exist_ok=True)

    data_list = gen_data_path_list_list(opt.Overfit.test_data_dir, opt.Train.batch_size)

    # maml_state_dict = torch.load(opj(maml_folder, 'model_maml.pth'), map_location='cpu')
    # torch.save(maml_state_dict, opj(overfit_folder, 'model_maml.pth'))
    
    for data_dir in data_list:

        data_name = data_dir[0].split('/')[-1].split('.')[0]
        data_extension = data_dir[0].split('/')[-1].split('.')[-1]

        orig_data = tifffile.imread(data_dir[0])
        if len(orig_data.shape) == 3:
            orig_data = orig_data[..., None]
        assert (
            len(orig_data.shape) == 4
        ), "Only DHWC data is allowed. Current data shape is {}.".format(orig_data.shape)

        # divide the original data into blocks
        d_origin, h_origin, w_origin, v_origin = orig_data.shape
        temp_dir = opj(opt.Train.train_data_dir, os.listdir(opt.Train.train_data_dir)[0])
        block_data = tifffile.imread(temp_dir)
        d_block, h_block, w_block, _ = block_data.shape
        level = int(d_origin / d_block)

        for i in range(level):
            for j in range(level):
                for k in range(level):
                    block_temp = copy.deepcopy(orig_data[i*d_block:(i+1)*d_block, j*h_block:(j+1)*h_block, k*w_block:(k+1)*w_block, :]) 
                    block_file_path =  opj(overfit_folder, data_name, "blocks")
                    if not os.path.exists(block_file_path):
                        os.makedirs(block_file_path)
                    block_path = opj(overfit_folder, data_name, "blocks", f"{i}_{j}_{k}" + '.' + data_extension)
                    tifffile.imwrite(block_path, block_temp)

                    dataset = FlattenedDataset(opt.Train.batch_size, opt.Normalize, True, data_path_list=[block_path])

                    for (data,sideinfos) in dataset:

                        data = data.to(device)
                        flattened_sampler = FlattenedSampler(data, opt.Train.sample_size, opt.Train.sample_count)

                        data_temp = gen_data_path_list_list(opt.Train.val_data_dir, 1)
                        ideal_network_size_bytes = os.path.getsize(data_temp[0][0]) / opt.Siren.compression_ratio
                        ideal_network_parameters_count = ideal_network_size_bytes / 4.0
                        n_network_features = Siren.calc_features(
                            ideal_network_parameters_count, opt.Siren.coords_channel,
                            opt.Siren.data_channel, opt.Siren.layers)
                        model = Siren(
                            opt.Siren.coords_channel, n_network_features, opt.Siren.layers-2,
                            opt.Siren.data_channel, True, opt.Siren.w0)
                        
                        root_path = opj(overfit_folder, data_name, 'train_from_meta')
                        if not os.path.exists(root_path):
                            os.makedirs(root_path)
                        if os.path.exists(opj(root_path, 'checkpoints', 'model_final.pth')):
                            print("Skipping ", root_path)
                            continue

                        # tblogger = SummaryWriter(root_path)
                        OmegaConf.save(opt, opj(root_path, "config.yaml"))

                        meta_phi = MAML(num_meta_steps=opt.Train.max_inner_steps, hypo_module=model,
                            loss=l2_loss, init_lr=opt.Train.lr_phi_inner,lr_type=opt.Train.lr_type)
                        meta_phi = meta_phi.to(device)

                        state_dict = torch.load(opj(maml_folder, 'maml_obj.pth'), map_location='cpu')
                        meta_phi.load_state_dict(state_dict, strict=True)
                        meta_phi.first_order = True
                        eval_model = copy.deepcopy(meta_phi.hypo_module)
                        num_maml_steps = opt.Train.max_inner_steps

                        for (sampled_coords, sampled_data) in flattened_sampler:
                            sampled_coords = sampled_coords.to(device)

                            context = {'x': sampled_coords, 'y': sampled_data}

                            # Specialize the model with the "generate_params" function.
                            fast_params, intermed_predictions = refine(meta_phi, context, num_maml_steps)

                            # Compute the final outputs.
                            model_output = meta_phi.hypo_module(sampled_coords, params=fast_params)
                            model_output = {'model_out': model_output, 'intermed_predictions': intermed_predictions,
                                            'fast_params': fast_params}
                            
                            fast_params_squeezed = {}
                            for name, param in fast_params.items():
                                if name == 'net.4.weight' :
                                    _, _, hidden_dim = param.shape
                                    fast_params_squeezed[name] = param.view(1,hidden_dim)
                                elif name == 'net.4.bias':
                                    fast_params_squeezed[name] = param.view(1)
                                else:
                                    fast_params_squeezed[name] = param.squeeze()
                            eval_model.load_state_dict(fast_params_squeezed)

                            # overfit from the eval_model
                            optim = torch.optim.Adam(eval_model.parameters(), lr=opt.Overfit.lr)

                            if not opt.Overfit.lr_static:
                                if opt.Overfit.warmup > 0:
                                    scheduler = ReduceLROnPlateauWithWarmup(optim, warmup_end_lr=opt.Overfit.lr, warmup_steps=opt.Overfit.warmup, 
                                                                            mode='min', factor=0.5, patience=opt.Overfit.patience, threshold=0.0001,
                                                                            threshold_mode='rel', cooldown=0, eps=1e-08, verbose=True)
                                else:
                                    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=opt.Overfit.patience, threshold=0.0001, 
                                                                threshold_mode='rel', cooldown=0, eps=1e-08, verbose=True)
                                
                            checkpoints_dir = os.path.join(root_path, 'checkpoints')
                            if not os.path.exists(checkpoints_dir):
                                os.makedirs(checkpoints_dir)
                            
                            for epoch in range(1, opt.Overfit.epoch + 1):
                                optim.zero_grad()
                                model_out = eval_model(sampled_coords)
                                loss = l2_loss(model_out, sampled_data)
                                loss.backward()
                                optim.step()
                                if not opt.Overfit.lr_static:
                                    scheduler.step(loss.item()) 
                                # if epoch % opt.Overfit.epoch_til_summary == 0 or epoch == 1:
                                #     tblogger.add_scalar(f"{data_name}/loss", loss.item(), epoch)
                                #     lr_cur = optim.state_dict()['param_groups'][0]['lr']
                                #     tblogger.add_scalar(f"{data_name}/lr", lr_cur, epoch)
                                if epoch % opt.Overfit.epoch_til_ckpt == 0 or epoch == 1:
                                    # save model and evaluate performance
                                    curr_epoch_dir = opj(checkpoints_dir, f'epochs_{epoch}')
                                    if not os.path.exists(curr_epoch_dir):
                                        os.makedirs(curr_epoch_dir)
                                    compressed_data_save_dir = opj(curr_epoch_dir, "compressed", f"{i}_{j}_{k}")
                                    if not os.path.exists(compressed_data_save_dir):
                                        os.makedirs(compressed_data_save_dir)
                                    model_parameters_save_dir = opj(compressed_data_save_dir, "network_parameters")
                                    save_model_(eval_model, model_parameters_save_dir, "cuda")

                                    # decompress data
                                    with torch.no_grad():
                                        data_shape = sideinfos['data_shape']
                                        n, data_channel, d, h, w = data_shape
                                        coordinates = torch.stack(
                                            torch.meshgrid(
                                                torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
                                                indexing="ij"), axis=-1)
                                        coordinates = coordinates.to(device)
                                        flattened_coords = rearrange(coordinates, "d h w c-> (d h w) c")
                                        flattened_decompressed_data = torch.zeros((d*h*w, 1), device="cuda")
                                        sample_size = opt.Overfit.sample_size
                                        n_batches = math.ceil(d*h*w / sample_size)
                                        for idx in range(n_batches):
                                            start_idx = idx * sample_size
                                            end_idx = min((idx + 1) * sample_size, d*h*w)
                                            flattened_decompressed_data[start_idx:end_idx] = eval_model(flattened_coords[start_idx:end_idx])
                                        decompressed_data = rearrange(flattened_decompressed_data,
                                                                    "(d h w) c -> d h w c", d=d, h=h, w=w)
                                        if decompressed_data.is_cuda:
                                            decompressed_data = decompressed_data.cpu()
                                        decompressed_data = invnormalize_data(decompressed_data, sideinfos,**opt.Normalize)

                                    # save decompressed data
                                    decompressed_data_save_dir = opj(curr_epoch_dir, "decompressed", f"{i}_{j}_{k}")
                                    if not os.path.exists(decompressed_data_save_dir):
                                        os.makedirs(decompressed_data_save_dir)
                                    decompressed_data_save_path = opj(decompressed_data_save_dir, data_name + "_decompressed" + '.' + data_extension)
                                    tifffile.imwrite(decompressed_data_save_path, decompressed_data)
      
        # eval performance
        tblogger = SummaryWriter(root_path)
        epoch_list = os.listdir(checkpoints_dir)
        epoch_list = sorted(epoch_list,key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)))
        for block_path in epoch_list:
            decompressed_whole_data = np.zeros([d_origin, h_origin, w_origin, v_origin])
            epochs = int(block_path.split('_')[-1])
            block_path = opj(checkpoints_dir, block_path)
            decompress_block_path = opj(block_path, "decompressed")
            for single_block_path in os.listdir(decompress_block_path):
                i, j, k = single_block_path.split('_')
                i = int(i)
                j = int(j)
                k = int(k)
                single_block_path = opj(decompress_block_path, single_block_path)
                block_name = os.listdir(single_block_path)[0]
                single_block_path = opj(single_block_path, block_name)
                single_block = tifffile.imread(single_block_path)
                decompressed_whole_data[i*d_block:(i+1)*d_block, j*h_block:(j+1)*h_block, k*w_block:(k+1)*w_block, :] = single_block

            decompressed_whole_data_save_path = opj(block_path, data_name + "_decompressed" + '.' + data_extension)
            tifffile.imwrite(decompressed_whole_data_save_path, decompressed_whole_data)
            mse, psnr = calc_mse_psnr(orig_data[..., 0], decompressed_whole_data[..., 0])
            ssim = calc_ssim(orig_data[..., 0], decompressed_whole_data[..., 0])

            tblogger.add_scalar(f"{data_name}/mse", mse, epochs)
            tblogger.add_scalar(f"{data_name}/psnr", psnr, epochs)
            tblogger.add_scalar(f"{data_name}/ssim", ssim, epochs)
            results = {k: None for k in EXPERIMENTAL_RESULTS_KEYS}
            results["algorithm_name"] = "Overfit_from_meta"
            results["epochs"] = epochs
            results["original_data_path"] = data_dir[0]
            results["config_path"] = os.path.abspath(args.p)
            results["decompressed_data_path"] = decompress_block_path
            results["data_name"] = data_name
            results["data_type"] = orig_data.dtype.name
            results["data_shape"] = orig_data.shape
            results["actual_ratio"] = os.path.getsize(data_dir[0]) / get_folder_size(opj(block_path, "compressed"))
            results["psnr"] = psnr
            results["mse"] = mse
            results["ssim"] = ssim
            csv_path = opj(root_path, "results.csv")
            if not os.path.exists(csv_path):
                f = open(csv_path, "a")
                csv_writer = csv.writer(f, dialect="excel")
                csv_writer.writerow(results.keys())
            row = [results[key] for key in results.keys()]
            csv_writer.writerow(row)
            f.flush()
        print(f"Finish {data_name}.{data_extension} train_data!", flush=True)
        tblogger.close()



if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    parser = argparse.ArgumentParser(description='overfit the original data')
    parser.add_argument('-p',type=str,default='Run_pipeline/train_data.yaml',help='yaml file path')
    parser.add_argument('-g',type=str,default='1',help='which gpu to use')
    args = parser.parse_args()
    opt: SingleTaskOpt = OmegaConf.load(args.p)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.g)

    # device
    device = 'cuda' if opt.Train.gpu else 'cpu'

    train_from_meta()