import os
from utils import constant

# first level:coarse classification
os.system('CUDA_VISIBLE_DEVICES=0 python train.py --input_dropout 0.4 --type coarse')

# second level
for coarse_name in constant.COARSE_TO_ID:
    os.system('CUDA_VISIBLE_DEVICES=0 python train.py --input_dropout 0.4 --type multi --coarse_name ' + coarse_name)

# output result
os.system('python result.py')