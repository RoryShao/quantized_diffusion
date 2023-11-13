export CUDA_VISIBLE_DEVICES="1"

# clib data to generate ckpt
# python scripts/txt2img_v1.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --quant_act --act_bit 8 --cali_st 25 --cali_batch_size 8 --cali_n 128 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt --outdir output_path/txt2img


# # clib data to generate ckpt
# python scripts/txt2img_v1.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 4 --cali_st 25 --cali_n 128  --cali_batch_size 8 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt --outdir output/txt2img_ckpt_w8a8/  

# # # clib data to generate ckpt
# python scripts/txt2img_v1.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 4 --cali_st 1 --cali_n 128  --cali_batch_size 8 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt --outdir output/txt2img_ckpt_w8a8/  


# python scripts/txt2img_v1.py --prompt  "a puppy wearing a hat" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --resume --outdir output_path/txt2img --cali_ckpt quantized_ckpt_path/sd_w4a8_ckpt.pth


# python scripts/txt2img_v1.py --prompt  "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --resume --outdir output_path/txt2img --cali_ckpt quantized_ckpt_path/sd_w4a8_ckpt.pth



#  python scripts/txt2img_quante.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq_sampling

 python scripts/txt2img_quante.py --prompt  "a cat wearing a hat" --plms --cond --ptq_sampling --weight_bit 4 --act_bit 8 --quant_mode qdiff 