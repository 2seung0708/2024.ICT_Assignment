# n_gpu=1
# cls='ape'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='eggbox'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 







# cls='benchvise'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='cam'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='can'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 

# cls='cat'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='driller'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='duck'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='glue'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='holepuncher'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='iron'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='lamp'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 


# cls='phone'
# tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar"
# python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py  --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose --gpu "2" 



##=========================================
## 
##=========================================
n_gpu=1

##  class 선택
cls_lst=("cat" ) # "duck" "iron" "glue"

for cls in "${cls_lst[@]}" 
do

    tst_mdl="./train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best*.pth.tar"
    
    ## 결과 저장할 경로 설정
    save_path="./train_log/linemod/shlee_log/${cls}"
    if [ ! -d $save_path ]
    then
        mkdir $save_path
        echo "생성완료"
    fi
    ##

    for ckp_path in $tst_mdl
    do
        echo $ckp_path 
        tt=`echo $ckp_path  | cut -d '/' -f6`
        tt=`echo $tt  | cut -d '.' -f1``echo $tt  | cut -d '.' -f2`
        echo $tt
        n_gpu+=1
        # OMP_NUM_THREADS=4 python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py --cls $cls -eval_net -checkpoint $ckp_path -test -test_pose --gpu '7'  > ${save_path}/${tt}_log.txt
        OMP_NUM_THREADS=4 python3 -m torch.distributed.launch --nproc_per_node=1 train_lm_shlee.py --custom True --cls $cls -eval_net -checkpoint $ckp_path -test -test_pose --gpu '5' #  > ${save_path}/${tt}_log.txt
    done
done
