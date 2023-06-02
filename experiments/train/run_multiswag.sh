# Run and evaluate MultiSWAG
DATAPATH=../../datasets/
DATASET=places365_3c #CIFAR100
MODEL='resnet50'

EPOCHS=300
SWAG_START=161
SWAG_SAMPLES=20

BASEDIR=ckpts/places365_multiswag_3c_resnet50

# places365_multiswag_smallvgg
# places365_multiswag_fullvgg 
# places365_singleswag_fullvgg

SWAG_RUNS=3

LR=0.01
WD=1e-4
SWAG_LR=0.01

CKPT_FILES=""

# seed=1
# python run_swag_single_fullvgg.py \
#     --swa_freq=1 \
#     --data_path=$DATAPATH \
#     --epochs=$EPOCHS \
#     --dataset=$DATASET \
#     --save_freq=$EPOCHS \
#     --model=$MODEL \
#     --lr_init=${LR} \
#     --wd=${WD} \
#     --swa \
#     --swa_start=$SWAG_START \
#     --swa_lr=${SWAG_LR} --cov_mat \
#     --dir=${BASEDIR}/swag_${seed} \
#     --seed=$seed 

for (( seed=1; seed <=$SWAG_RUNS; seed++ ))
do
    # python run_swag.py --data_path=$DATAPATH --epochs=$EPOCHS --dataset=$DATASET --save_freq=$EPOCHS \
    #   --model=$MODEL --lr_init=${LR} --wd=${WD} --swag --swag_start=$SWAG_START --swag_lr=${SWAG_LR} --cov_mat --use_test \
    #   --dir=${BASEDIR}/swag_${seed} --seed=$seed 
    python run_swag_single_fullvgg.py --data_path=$DATAPATH --epochs=$EPOCHS --dataset=$DATASET --save_freq=$EPOCHS \
      --model=$MODEL --lr_init=${LR} --wd=${WD} --swa --swa_start=$SWAG_START --swa_lr=${SWAG_LR} --cov_mat \
      --dir=${BASEDIR}/swag_${seed} --seed=$seed 
    CKPT_FILES+=" "${BASEDIR}/swag_${seed}/swag-${EPOCHS}.pt
done


# for ((model_id=1; model_id<=$SWAG_RUNS; model_id++ ))
# do
#     CKPT_FILES=" "${BASEDIR}/swag_${model_id}/swag-${EPOCHS}.pt
#     python3 eval_multiswag_single_fullvgg.py \
#         --data_path=$DATAPATH \
#         --dataset=$DATASET \
#         --model=$MODEL \
#         --model_id=$model_id \
#         --use_test \
#         --swag_ckpts \
#         --swag_samples=$SWAG_SAMPLES \
#         --swag_ckpts${CKPT_FILES}  \
#         --savedir=$BASEDIR/multiswag/
# done

python3 eval_multiswag_single_fullvgg.py --data_path=$DATAPATH --dataset=$DATASET --model=$MODEL --use_test --swag_ckpts \
  --swag_samples=$SWAG_SAMPLES --swag_ckpts${CKPT_FILES}  --savedir=$BASEDIR/multiswag/



