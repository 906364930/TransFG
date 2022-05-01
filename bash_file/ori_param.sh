CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch \
--nproc_per_node=4 train.py \
--dataset CUB_200_2011 \
--split overlap \
--num_steps 10000 \
--train_batch_size 16 \
--fp16 \
--name pruned_qkv_fixed \

