export CUDA_VISIBLE_DEVICES="1"

# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 -l output_lsun --cali_ckpt quantized_ckpt_path/church_w4a8_ckpt.pth  --cali_data_path clib_data/church_sample1020_allst.pt --cali_n 1000 --cali_st 2


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 8  --resume -l output_lsun --cali_ckpt quantized_ckpt_path/church_w4a8_ckpt.pth  --cali_data_path clib_data/church_sample1020_allst.pt --cali_n 1000 --cali_st 2


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 --quant_act --act_bit 8  --resume -l output_lsun --cali_ckpt quantized_ckpt_path/church_w4a8_ckpt.pth  --cali_data_path clib_data/church_sample1020_allst.pt --cali_n 1000 --cali_st 2


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 --quant_act --act_bit 8  --resume -l output_lsun --cali_ckpt quantized_ckpt_path/church_w4a8_ckpt.pth  --cali_data_path clib_data/church_sample1020_allst.pt --cali_n 1000 --cali_st 2


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 50000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 --quant_act --act_bit 4  --resume -l output_lsun  --cali_data_path clib_data/church_sample1020_allst.pt --cali_n 1000 --cali_st 2


# Calibration quanted model to generate quanted W8A8.pt
python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_churches256/model.ckpt -n 5000 --batch_size 10 -c 400 -e 0.0 --seed 40 --ptq --weight_bit 8 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 8 --cali_data_path clib_data/church_sample1020_allst.pt -l output_lsun_ckpt_w8a8
