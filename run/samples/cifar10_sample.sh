export CUDA_VISIBLE_DEVICES="1"

# 4/8-bit weights-only
python scripts/sample_diffusion_ddim_v1.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --weight_bit 4 --quant_mode qdiff --split --resume -l output_cifar10 --cali_ckpt output_cifar10/samples/2023-10-15-16-35-32/ckpt.pth