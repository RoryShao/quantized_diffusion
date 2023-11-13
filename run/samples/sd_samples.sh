export CUDA_VISIBLE_DEVICES="2"

python run_scripts/txt2img.sh --prompt  "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --resume --outdir output_path/txt2img --cali_ckpt quantized_ckpt_path/sd_w4a8_ckpt.pth

# python run_scripts/txt2img.sh --prompt  "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --quant_act --no_grad_ckpt --split --n_samples 5 --resume --outdir output_path/txt2img --cali_ckpt output_path/txt2img/2023-10-16-01-47-31/ckpt.pth


# python run_scripts/txt2img.sh --prompt  "a puppet wearing a hat" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --outdir output_path/txt2img --cali_ckpt  output_path/txt2img/2023-10-16-01-47-31/ckpt.pth



# # clib data to generate ckpt
# python run_scripts/txt2img.sh --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 4 --cali_st 2 --cali_n 128  --cali_batch_size 64 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt --outdir output/txt2img_ckpt_w8a8/ 