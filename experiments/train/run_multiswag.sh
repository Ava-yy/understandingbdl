# Run and evaluate MultiSWAG
DATAPATH=../../datasets/
DATASET=places365_3c #CIFAR100
MODEL='vgg16'
EPOCHS=300
BASEDIR=ckpts/places365_multiswag_3c

# places365_multiswag_smallvgg
# places365_multiswag_fullvgg 
# places365_singleswag_fullvgg

SWAG_RUNS=3

LR=0.01
WD=1e-4
SWAG_START=1
SWAG_LR=0.02
SWAG_SAMPLES=20
echo ${WD}

CKPT_FILES=""

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


python3 eval_multiswag_single_fullvgg.py --data_path=$DATAPATH --dataset=$DATASET --model=$MODEL --use_test --swag_ckpts \
  --swag_samples=$SWAG_SAMPLES --swag_ckpts${CKPT_FILES}  --savedir=$BASEDIR/multiswag/
