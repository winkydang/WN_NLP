#python3 -m torch.distributed.launch --nproc_per_node=3  \
#train.py --gpu '1,2,3' \

torchrun train.py \
--bs 32 \
--n_epochs 21 \
#--ckpt_path './tmp/save/train/model-4.ckpt'
#--bert_path './tmp/save/train/model-4.ckpt'



