CUDA_VISIBLE_DEVICES=0,1
python train.py \
    --batch_size 32 \
    --max_iter 1000000 \
    --n_flow 32 \
    --n_block 6 \
    --n_bits 5 \
    --lr 1e-5 \
    --img_size 128 \
    --temp 0.7 \
    --gradient_accumulation_steps 2 \
    --checkpoint_iter 100000 \
    --generation_iter 100 \
    --dataset_name CELEBA \
    --network_config 0 \
    --cls_loss_weight 0.0