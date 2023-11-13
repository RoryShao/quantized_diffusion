export CUDA_VISIBLE_DEVICES="0"

python scripts/training_diffusion_ldm_quante_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41  --weight_bit 8 --act_bit 8 -l output/lsun --wwq --waq --act_quant --data_dir /data/rrshao/datasets/bedroom_train/ --calib_data noise_data  --ptq #--use_adaround 


python scripts/training_diffusion_ldm_quante_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41  --weight_bit 4 --act_bit 8 -l output/lsun --wwq --waq --act_quant --data_dir /data/rrshao/datasets/bedroom_train/ --calib_data noise_data  --ptq #--use_adaround 


python scripts/training_diffusion_ldm_quante_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41  --weight_bit 4 --act_bit 4 -l output/lsun --wwq --waq --act_quant --data_dir /data/rrshao/datasets/bedroom_train/ --calib_data noise_data  --ptq #--use_adaround 


python scripts/training_diffusion_ldm_quante_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41  --weight_bit 2 --act_bit 2 -l output/lsun --wwq --waq --act_quant --data_dir /data/rrshao/datasets/bedroom_train/ --calib_data noise_data  --ptq #--use_adaround  

python scripts/dm_quante_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41  --weight_bit 4 --act_bit 8 -l output/lsun --wwq --waq --act_quant --data_dir /data/rrshao/datasets/bedroom_train/ --calib_data noise_data  --ptq --use_adaround  