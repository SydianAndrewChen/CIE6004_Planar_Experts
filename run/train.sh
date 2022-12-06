# Train teacher model
# python teacher_train.py --config-name $1
# Distill experts from teacher and finetune
CUDA_VISIBLE_DEVICES=0,1 python experts_train.py --config-name $1

