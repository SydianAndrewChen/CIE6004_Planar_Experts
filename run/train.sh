# Train teacher model
# python teacher_train.py --config-name $1
# Distill experts from teacher and finetune
# CUDA_VISIBLE_DEVICES=0 python experts_train.py --config-name $1
CUDA_VISIBLE_DEVICES=0 python experts_train_wo_teacher.py --config-name $1

