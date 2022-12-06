# Render images
CUDA_VISIBLE_DEVICES=1 python experts_test.py test.mode=render model.accelerate.bake=False --config-name $1
# Evaluate metrics
# python -m mnh.metric -rewrite output_images/$1/experts/color/valid/ 