export CUDA_VISIBLE_DEVICES="0"

# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 8  --act_bit 8 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data noise_data --ptq  --calib_model # --use_adaround 


# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 4 --act_bit 4 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data noise_data --ptq --calib_model #  --use_adaround 


# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 2 --act_bit 2 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data noise_data  --ptq  --calib_model # --use_adaround

# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 4 --act_bit 8 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data noise_data  --ptq  --calib_model # --use_adaround

# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 4 --act_bit 8 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data noise_data  --ptq  --calib_model --use_adaround

python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 4 --act_bit 8 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data raw_data  --ptq  --calib_model --use_adaround


# python scripts/ldm_quante_churches.py -r models/ldm/lsun_churches256/model.ckpt -n 1000 --batch_size 64 -c 200 -e 1.0 --seed 41 --weight_bit 4 --act_bit 8 -l output/lsun  --wwq --waq --act_quant --data_dir /data/rrshao/datasets/lsun/church_outdoor_train/ --calib_data timestep_aware  --ptq  --calib_model --use_adaround
