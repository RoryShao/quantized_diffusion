export CUDA_VISIBLE_DEVICES="3"

python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_beds256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 8 --a_sym --a_min_max --running_stat -l output_beds256/W4A8/ --cali_data_path clib_data/bedroom_sample2040_allst.pt 


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_beds256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 8 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 8 --a_sym --a_min_max --running_stat -l output_beds256/W8A8/ --cali_data_path clib_data/bedroom_sample2040_allst.pt 


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_beds256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 8 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 4 --a_sym --a_min_max --running_stat -l output_beds256/W8A4/ --cali_data_path clib_data/bedroom_sample2040_allst.pt 


# python scripts/sample_diffusion_ldm_v1.py -r models/ldm/lsun_beds256/model.ckpt -n 5000 --batch_size 64 -c 200 -e 1.0 --seed 41 --ptq --weight_bit 4 --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit 4 --a_sym --a_min_max --running_stat -l output_beds256/W4A4/ --cali_data_path clib_data/bedroom_sample2040_allst.pt 