import argparse, os, sys, glob, datetime, yaml
import logging
import torch
import time
import numpy as np
from tqdm import trange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import torch.nn as nn
from utils.image_datasets import load_data

import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
import warnings
warnings.filterwarnings('ignore')
from QDrop.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)
# from datasets.imagenet64 import build_imagenet_data


CUDA_VISIBLE_DEVICES="1"


logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):


    log = dict()

    # shape = [batch_size,
    #          model.model.diffusion_model.in_channels,
    #          model.model.diffusion_model.image_size,
    #          model.model.diffusion_model.image_size]
    
    shape = [64, 4, 32, 32] 

    # with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape, eta=eta)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape, eta=eta)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, dpm=dpm)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def generate_t(args, t_mode, num_calib, diffusion, device):
    if t_mode == "random":
        # t = torch.randint(0, diffusion.num_timesteps, [num_calib], device=device)
        t = torch.randint(0, int(diffusion.num_timesteps*0.8), [num_calib], device=device)
        print(t.shape)
        print(t)
    elif t_mode == "uniform":
        t = torch.linspace(
            0, diffusion.num_timesteps, num_calib, device=device
        ).round()
    else:
        raise NotImplementedError
    return t.clamp(0, diffusion.num_timesteps - 1)


def raw_calib_data_generator(
    args, num_calib, device, t_mode, diffusion, class_cond=True
):
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=128,
        image_size=256, #args.image_size,
        class_cond=class_cond,
    )
    # TODO: sample a batch images
    calib_data, cls = next(loader)
    calib_data = calib_data.to(device)
    t = generate_t(args, t_mode, num_calib, diffusion, device)
    # t = diffusion._scale_timesteps(t)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t

'''
    construct the model to 
'''
def obtain_calib_data(opt):
    cali_data=None
    if opt.calib_data == 'raw_data':
        sample_cali_data = raw_calib_data_generator( opt, opt.num_calib, "cuda", opt.calib_t_mode, model, class_cond=False)
        encoder_posterior = model.encode_first_stage(sample_cali_data[0])
        # TODO: timestep --> laten embedding, learn LatentDiffusion of the ddpm
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        cali_data = (z, sample_cali_data[1])

    elif opt.calib_data == 'noise_data':
        image_size = config.model.params.image_size
        channels = config.model.params.channels
        cali_data = (torch.randn(256, channels, image_size, image_size), torch.randint(0, 1000, (256,)))
    
    elif opt.calib_data == 'sample_data':     
        shape = [256, 4, 32, 32]
        sample, intermediates = convsample_ddim(model,  steps=opt.custom_steps, shape=shape, eta=1.0)
        cali_data = (sample, torch.randint(0, 1000, (256,)))

    elif opt.calib_data == 'timestep_aware':     
    # TODO: timestep --> laten embedding
        shape = [256, 4, 32, 32]
        t_label = 1
        cali_dataset = torch.randn(1, 4, 32, 32).cuda()
        # for i in range(1, opt.custom_steps):            
        for i in range(1, 500//25):            
            # if i % 25 == 0:
            t_label = t_label + 1
            sample, intermediates = convsample_ddim(model,  steps=500, shape=shape, eta=1.0) # opt.custom_steps = 500
            cali_dataset = torch.cat((cali_dataset,sample), dim=0)
        # TODO : t uses as the conditional label 
        random_label = torch.randint(0, 1000, (cali_dataset.shape[0],))
        cali_data = (cali_dataset, random_label)

    return cali_data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        nargs="?",
        help="size of the calibration dataset",
        default=1024
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="?",
        help="the bs",
        default=256
    )    
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--data_dir",  type=str, 
        help="path to data."
    )

    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )

    parser.add_argument(
        "--calib_data", type=str, default="raw_data", 
        choices=["raw_data", "noise_data", "sample_data", "timestep_aware"], 
        help="quantization mode to use"
    )
    # qdiff specific configs
    # parser.add_argument(
    #     "--cali_ckpt", type=str,
    #     help="path for calibrated model ckpt"
    # )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--workers",type=int, default=8,
        help="attn softmax activation bit"
    )    
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--disable_8bit_head_stem", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )

    parser.add_argument(
        "--awq",
        action="store_true",
        help="weight_quant for input in activation reconstruction",
    )
    parser.add_argument(
        "--aaq",
        action="store_true",
        help="act_quant for input in activation reconstruction",
    )
   
    parser.add_argument(
        "--init_wmode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for weight",
    )
    parser.add_argument(
        "--init_amode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for activation",
    )
    parser.add_argument(
        "--act_quant", action="store_true", help="apply activation quantization"
    )
    parser.add_argument(
        "--calib_t_mode",
        default="random",
        type=str,
        choices=["random", "1", "-1", "mean", "uniform" , 'manual' ,'normal' ,'poisson'],
    )
    # weight calibration parameters
    # parser.add_argument('--num_calib', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')
    parser.add_argument('--wwq', action='store_true', help='weight_quant for input in weight reconstruction')
    parser.add_argument('--waq', action='store_true', help='act_quant for input in weight reconstruction')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument("--use_adaround", action="store_true")
    parser.add_argument("--calib_model", action="store_true")

     # order parameters
    parser.add_argument('--order', default='before', type=str, choices=['before', 'after', 'together'], help='order about activation compare to weight')
    parser.add_argument('--prob', default=1.0, type=float)
    parser.add_argument('--input_prob', default=1.0, type=float)
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def get_train_samples(train_loader, num_calib):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_calib:
            break
    return torch.cat(train_data, dim=0)[:num_calib], torch.cat(target, dim=0)[:num_calib]


def random_calib_data_generator(shape, num_calib, device,class_cond=True):
    calib_data = []
    for batch in range(num_calib):
        img = torch.randn(*shape, device=device)
        calib_data.append(img)
    t = torch.tensor([1] * num_calib, device=device)  # TODO timestep gen
    if class_cond:
        cls = torch.tensor([1] * num_calib, device=device).long()  # TODO class gen
        return torch.cat(calib_data, dim=0), t, cls
    else:
        return torch.cat(calib_data, dim=0), t


def run_sample(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=custom_steps,
                                                eta=eta, dpm=dpm)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        # shape_str = "x".join([str(x) for x in all_img.shape])
        # nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        # np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def quant_model(opt, config, model, cali_data, logger):
    diffusion = model.model.diffusion_model
    # diffusion = model

    wq_params = {
        "n_bits": opt.weight_bit,
        "channel_wise": True,
        "scale_method": opt.init_wmode,
        "symmetric": True,
    }
    aq_params = {
        "n_bits":  opt.act_bit, 
        "channel_wise": False,
        "scale_method": opt.init_amode,
        "leaf_param": True,
        "prob": opt.prob,
        "symmetric": True,
    }
    # logger.info('before quanted')
    # state_dict = diffusion.state_dict()
    # logger.info(state_dict)
    # # 遍历打印每个参数的名称和形状
    # for name, param in state_dict.items():
    #     logger.info(name, param.shape)

    qnn = QuantModel(model=diffusion, weight_quant_params=wq_params, act_quant_params=aq_params)
        # ,sm_abit=opt.sm_abit)
    qnn.cuda()
    qnn.eval()
    logger.info(f"model arch:{qnn}")
    """init weight quantizer"""
    set_weight_quantize_params(qnn)

    # logger.info('after quanted')
    # state_dict = qnn.state_dict()
    # logger.info(state_dict)
    # # 遍历打印每个参数的名称和形状
    # for name, param in state_dict.items():
    #     logger.info(name, param.shape)

    # if not opt.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     qnn.set_first_last_layer_to_8bit()

    # qnn.disable_network_output_quantization()
    # print('check the model!')
    # print(qnn)
    
    if not opt.calib_model:
        print('directly return the quanted model!')
        return qnn
    # opt.batch_size = cali_dataset.shape[0]
    assert opt.wwq is True
    kwargs = dict(cali_data=cali_data, iters=opt.iters_w, weight=opt.weight,
                  b_range=(opt.b_start, opt.b_end), warmup=opt.warmup, opt_mode='mse',
                  wwq=opt.wwq, waq=opt.waq, order=opt.order, act_quant=opt.act_quant,
                  lr=opt.lr, input_prob=opt.input_prob, keep_gpu=not opt.keep_cpu)
    
    if opt.act_quant and opt.order == 'before' and opt.awq is False:
        '''Case 2'''
        set_act_quantize_params(qnn, cali_data=cali_data, awq=opt.awq, order=opt.order)
        # resume_cali_model(qnn, ckpt, cali_data, opt.quant_act, "qdiff", cond=False)
    
    if not opt.use_adaround:  # Do not use adaround to 
        print('setting')
        # cali_data = cali_data.detach()
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=opt.awq, order=opt.order
        )
        print('setting1111111')
        qnn.set_quant_state(weight_quant=True, act_quant=opt.act_quant)        
        return qnn
    else:
        def set_weight_act_quantize_params(module):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, module, **kwargs)
            else:
                raise NotImplementedError

        def recon_model(model: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print("Reconstruction for layer {}".format(name))
                    set_weight_act_quantize_params(module)
                elif isinstance(module, BaseQuantBlock):
                    print("Reconstruction for block {}".format(name))
                    set_weight_act_quantize_params(module)
                else:
                    recon_model(module)

        # Start calibration
        recon_model(qnn)

        if opt.act_quant and opt.order == "after" and opt.waq is False:
            """Case 1"""
            set_act_quantize_params(
                qnn, cali_data=cali_data, awq=opt.awq, order=opt.order
            )

        qnn.set_quant_state(weight_quant=True, act_quant=opt.act_quant)
        return qnn
    

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None


    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    print(config)

    logger.info(parser)
    logger.info(opt)
    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"model arch:{model}")
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)
    # print(model.model)
    
    # fix random seed
    seed_everything(opt.seed)
    
    if opt.ptq:
        # step1: Constructing the calibration dataset. 
        cali_data = obtain_calib_data(opt=opt)
        # step2: Quantify and calibrate the model.
        print('ptq start!')
        qnn = quant_model(opt, config, model, cali_data, logger)
        logger.info("done.")
        model.model.diffusion_model = qnn
        # model = qnn

        run_sample(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm)
        torch.save(qnn.state_dict(), logdir+'/'+"quanted_Unet_w{}a{}_ptq_{}.pt".format(opt.weight_bit, opt.act_bit, opt.ptq))     
        # torch.save(model.state_dict(), logdir+'/'+"quante_model_w{}a{}_ptq_{}.pt".format(opt.weight_bit, opt.act_bit, opt.ptq))     
    else:
        run_sample(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm)
        # torch.save(model.state_dict(), logdir+'/'+"model.pt")