import os
from utils import constant

DEVICES = str(0)
# 1 grained classification
os.system(
    'CUDA_VISIBLE_DEVICES=' + DEVICES + ' python train.py --input_dropout 0.3 --type grained --res_dir result/grained/ --data_dir dataset --save_dir saved_models/grained/')
os.system('rm -f saved_models/grained/checkpoint_epoch_*')

# 2 first level:coarse classification
os.system(
    'CUDA_VISIBLE_DEVICES=' + DEVICES + ' python train.py --input_dropout 0.3 --type coarse --res_dir result/coarse/ --data_dir dataset/coarse --save_dir saved_models/coarse/')
os.system('rm -f saved_models/coarse/checkpoint_epoch_*')

# 3 second level
for coarse_name in constant.COARSE_INTO_MULTI:
    os.system(
        'CUDA_VISIBLE_DEVICES=' + DEVICES + ' python train.py --input_dropout 0.4 --type multi --data_dir dataset/multi/' + coarse_name + ' --coarse_name ' + coarse_name + ' --save_dir saved_models/multi/' + coarse_name + ' --res_dir result/multi/' + coarse_name)
    os.system('rm -f saved_models/multi/' + coarse_name + '/checkpoint_epoch_*')
    os.system('CUDA_VISIBLE_DEVICES=' + DEVICES + ' python eval.py --data_dir dataset/multi/' + coarse_name + '/eval --test_filename test.tsv --model_dir saved_models/multi/' + coarse_name + '/best_model.pt')

# 4 output result
os.system('python result.py')