export PYTHONPATH=$PYTHONPATH:$(pwd)
data_path="/Path/to/EgoAfford"
gpus=1

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_multi.py \
    --model_path '/Path/to/ckpt' \
    --image_dir $data_path \
     --vis