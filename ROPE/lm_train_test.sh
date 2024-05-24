#!/bin/bash


## shlee 

# ## ------------------------------------------------------------------------------------------------------------------------
# #### 1. 필요한 파일 만들기
# ## ------------------------------------------------------------------------------------------------------------------------

# # # all class
# # class_list=("ape" "benchvise" "cam" "can" "cat" "driller" "duck" "eggbox" "glue" "holepuncher" "iron" "lamp" "phone")

# # v3. all class=with cup/bow
# class_list=("ape" "benchvise" "bowl" "cam" "can" "cat" "cup" "driller" "duck" "eggbox" "glue" "holepuncher" "iron" "lamp" "phone")


# idv=1
# # for문을 사용하여 문자열 리스트 반복
# for ((index=0; index<${#class_list[@]}; index++))
# do
#   class="${class_list[index]}"
#   # echo "Index $index: $class"
#   let obj_i=$index+$idv
#   # echo "$obj_i"

#   # echo "/home/shlee/workspace/6D_PoseEstimation/dataset/bop_challenge/lm/models/obj_$(printf "%06d" "$obj_i").ply"
  
#   # python gen_pts3d_fps.py --pts3d  --ply "/home/shlee/workspace/6D_PoseEstimation/dataset/bop_challenge/lm/models/obj_$(printf "%06d" "$obj_i").ply" --save_path "/home/shlee/workspace/6D_PoseEstimation/ROPE/dataset/obj$(printf "%02d" "$obj_i").mat"
  
#   #v3
#   echo  "/home/shlee/workspace/6D_PoseEstimation/dataset/LINEMOD_ORIG/${class_list[index]}/mesh.ply" 
#   python gen_pts3d_fps.py --pts3d  --ply "/home/shlee/workspace/6D_PoseEstimation/dataset/LINEMOD_ORIG/${class_list[index]}/mesh.ply" --save_path "/home/shlee/workspace/6D_PoseEstimation/ROPE/dataset/obj$(printf "%02d" "$obj_i").mat"

# done
# # ------------------------------------------------------------------------------------------------------------------------


## ------------------------------------------------------------------------------------------------------------------------
#### 2. 학습 및 평가
####    - 현재 코드는 0~7번 GPU를 모두 사용하기 위한 코드!
####    - glue는 이미 학습이 완료되어 사용하지 않음!
## ------------------------------------------------------------------------------------------------------------------------

# # v1. all class
# class_list=("ape" "benchvise" "cam" "can" "cat" "driller" "duck" "eggbox" "glue" "holepuncher" "iron" "lamp" "phone")

# # v2. train class
# class_list=("ape" "benchvise" "cam" "can" "cat" "driller" "duck" "eggbox" "holepuncher" "iron" "lamp" "phone")

# # v3. all class=with cup/bow
# class_list=("ape" "benchvise" "bowl" "cam" "can" "cat" "cup" "driller" "duck" "eggbox" "glue" "holepuncher" "iron" "lamp" "phone")

## 클래스 지정
class_list=("cat") 

# hr_ckpt_list=("hrnet_w32-36af842e_imagenet" "pose_hrnet_w32_256x256_mpii" "pose_hrnet_w32_384x288_posecoco" )
hr_ckpt="hrnet_w32-36af842e_imagenet"

exp_name="2405_LM" 

idx=0
for ((index=0; index<${#class_list[@]}; index++))
do
  class="${class_list[index]}"
  if [ "$class" == "bowl" ]
  then
    # echo 
    continue
  fi

  if [ "$class" == "cup" ]
  then
    # echo $class
    continue
  fi
  echo $class
  gpu_id=3

  exp_name="2405_train180_250epoch_${hr_ckpt}"
  echo $exp_name

  # file_path="./outputs/$class/${exp_name}_result.mat"
  # echo $file_path
  
  ## 합성 데이터 사용 O
  ## test만 할 경우 option:  `--test-only`  `--resume` 옵션 사용하기
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu_id python main_lm_shlee.py --cfg cfg_only_LM.yaml --obj $class --log_name $exp_name  --hr_ckpt "./model/${hr_ckpt}.pth" --test-only --resum 
  

  # ## 학습이 종료된 경우 학습 X 아닌 경우 추가학습하도록 작성한 코드 (23.12.21)
  # file_path="./outputs/$class/${exp_name}_result.mat"
  # if [ -e "$file_path" ]; then
  #     echo "$file_path 파일이 존재합니다."
  # else
  #     echo "$file_path 파일이 존재하지 않습니다."
  #     let idx=$idx+1
  #     let gpuid=$idx%4+4
  #     echo "Class: $class GPU_ID: $gpuid" >> ./outputs/class_per_gpu_id.txt
  #     OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpuid python main_lm.py --cfg cfg.yaml --obj $class --log_name $exp_name --resume&
  # fi

done

## ------------------------------------------------------------------------------------------------------------------------
