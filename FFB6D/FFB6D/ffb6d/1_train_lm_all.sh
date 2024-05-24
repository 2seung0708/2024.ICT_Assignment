#!/bin/sh

# n_gpu=4
# cls='ape'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='eggbox'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='benchvise'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# n_gpu=2
# cls='cam'
# # OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "4,5" 



# cls='can'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 

# cls='cat'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='driller'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='duck'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='glue'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='holepuncher'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='iron'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='lamp'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


# cls='phone'
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "1,2,5,6" 


##==================================
## for문으로 학습하기
##----------------------------------
### 전체 class
### cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
### iron: 다리미 can: 물뿌리개 
##==================================

n_gpu=1

## 학습할 class 선택
cls_lst=("cat") # ('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')

for cls in "${cls_lst[@]}" 
do
    # echo ${cls_lst[idx]}
    echo $cls
    ## 사용할 GPU 설정하기 gup="_,_"
    OMP_NUM_THREADS=6 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "3"
done

# # 디버깅용
# cls="phone"
# OMP_NUM_THREADS=8 python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --gpu "4,5" 