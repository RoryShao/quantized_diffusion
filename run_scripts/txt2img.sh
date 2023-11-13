export CUDA_VISIBLE_DEVICES="0"

python scripts/txt2img_quante.py --prompt  "a cat wearing a hat" --plms --cond --ptq --weight_bit 4 --act_bit 4 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --outdir output_path/txt2img --wwq --waq --calib_data timestep_aware  --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt  --cali_st 2 --cali_n 8

python scripts/txt2img_quante.py --prompt  "a cat wearing a hat" --plms --cond --ptq --weight_bit 4 --act_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --outdir output_path/txt2img --wwq --waq --calib_data timestep_aware  --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt  --cali_st 2 --cali_n 8 

python scripts/txt2img_quante.py --prompt  "a cat wearing a hat" --plms --cond --ptq --weight_bit 8 --act_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --outdir output_path/txt2img --wwq --waq --calib_data timestep_aware  --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt  --cali_st 2 --cali_n 8 --use_adaround 

python scripts/txt2img_quante.py --prompt  "a cat wearing a hat" --plms --cond --ptq --weight_bit 4 --act_bit 8 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --outdir output_path/txt2img --wwq --waq --calib_data timestep_aware  --cali_data_path clib_data/sd_coco-s75_sample1024_allst.pt  --cali_st 2 --cali_n 8 --use_adaround 
