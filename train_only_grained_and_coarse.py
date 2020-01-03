import os

DEVICES = str(0)
# 1 grained classification
os.system(
    'CUDA_VISIBLE_DEVICES=' + DEVICES + ' python train.py --input_dropout 0.3 --type grained --res_dir result/grained/ --data_dir dataset --save_dir saved_models/grained/')
os.system('rm -f saved_models/grained/checkpoint_epoch_*')

# 2 coarse classification
os.system(
    'CUDA_VISIBLE_DEVICES=' + DEVICES + ' python train.py --input_dropout 0.3 --type coarse --res_dir result/coarse/ --data_dir dataset/coarse --save_dir saved_models/coarse/')
os.system('rm -f saved_models/coarse/checkpoint_epoch_*')

# 3 result
os.system("pathon result/select_top5.py")