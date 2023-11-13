import argparse, os, datetime, yaml
import logging
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import torch.nn as nn
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


from QDrop.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
    data_utils
)

# solve the warnings 'Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used when initializing CLIPTextModel'
import transformers
transformers.logging.set_verbosity_error()
from keras import backend 

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def sample_image(opt, outpath, model, sampler, device):
    logging.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        logging.info(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


def obtain_calib_data(opt):
    cali_data = None
    if opt.calib_data == 'raw_data':
        cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
    elif opt.calib_data == 'timestep_aware':    
        load_data = torch.load(opt.cali_data_path)
        cali_data = data_utils.get_train_samples(args=opt, sample_data=load_data, custom_steps=opt.ddim_steps)
    return cali_data

def quant_model(opt, sampler, model,  cali_data, outpath, device):
    if opt.quant_mode == 'qdiff':
        wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
        aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  opt.quant_act}

        qnn = QuantModel(model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params) 

        qnn.cuda()
        qnn.eval()

        assert opt.wwq is True
        kwargs = dict(cali_data=cali_data, iters=opt.iters_w, weight=opt.weight,
                    b_range=(opt.b_start, opt.b_end), warmup=opt.warmup, opt_mode='mse',
                    wwq=opt.wwq, waq=opt.waq, order=opt.order, act_quant=opt.act_quant,
                    lr=opt.lr, input_prob=opt.input_prob, keep_gpu=not opt.keep_cpu)
        
        if opt.act_quant and opt.order == 'before' and opt.awq is False:
            '''Case 2'''
            set_act_quantize_params(qnn, cali_data=cali_data, awq=opt.awq, order=opt.order)
        
        """init weight quantizer"""
        set_weight_quantize_params(qnn)
        if not opt.use_adaround:
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
    

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--calib_data", type=str, default="raw_data", 
        choices=["raw_data", "noise_data", "sample_data", "timestep_aware"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--ptq_sampling", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
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
        "--quant_mode", type=str, default="symmetric", 
        choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--disable_8bit_head_stem", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')
    parser.add_argument('--wwq', action='store_true', help='weight_quant for input in weight reconstruction')
    parser.add_argument('--waq', action='store_true', help='act_quant for input in weight reconstruction')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument("--use_adaround", action="store_true")
    parser.add_argument('--order', default='before', type=str, choices=['before', 'after', 'together'], help='order about activation compare to weight')
    parser.add_argument(
        "--act_quant", action="store_true", help="apply activation quantization"
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
   
    parser.add_argument('--prob', default=1.0, type=float)
    parser.add_argument('--input_prob', default=1.0, type=float)

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')

    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    # parser.add_argument("--calib_model", action="store_true")
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
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

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # logger.info(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    assert(opt.cond)
    if opt.ptq:
        # step1: Constructing the calibration dataset. 
        cali_data = obtain_calib_data(opt=opt)
        qnn = quant_model(opt, sampler, model, cali_data, outpath, device)
        logger.info("done.")
        # model.model.diffusion_model = qnn
        sampler.model.model.diffusion_model = qnn
        sample_image(opt, outpath, model, sampler, device)
        torch.save(qnn.state_dict(), outpath+'/'+"quanted_Unet_w{}a{}_ptq_{}.pt".format(opt.weight_bit, opt.act_bit, opt.ptq))  
        torch.save(model.state_dict(), outpath+'/'+"quante_model_w{}a{}_ptq_{}.pt".format(opt.weight_bit, opt.act_bit, opt.ptq)) 
    if opt.ptq_sampling:
        wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
        aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  opt.quant_act}

        qnn = QuantModel(model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params)
        qnn.cuda()
        qnn.eval()

        qnn.load_state_dict(torch.load('output_path/txt2img/2023-10-22-13-27-17/quante_model_w4a8_ptq_True.pt', map_location='cpu'))
        # sampler.model.model.diffusion_model = qnn
        # sample_image(opt, outpath, model, sampler, device)
    else:
        sample_image(opt, outpath, model, sampler, device)
        # torch.save(model.state_dict(), outpath+'/'+"quante_model_w{}a{}_ptq_{}.pt".format(opt.weight_bit, opt.act_bit, opt.ptq))



if __name__ == "__main__":
    main()
