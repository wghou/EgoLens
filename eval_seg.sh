export PYTHONPATH=$PYTHONPATH:$(pwd)
model_name="/Path/to/ckpt"
data_path="/Path/to/EgoAfford"
gpus=1

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_seg_multi.py \
    --model_path $model_name \
    --image_dir $data_path \
    --type "gpt" --vis

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_seg_multi.py \
    --model_path $model_name \
    --image_dir $data_path \
    --type "gpt_external" --vis

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_seg_multi.py \
    --model_path $model_name \
    --image_dir $data_path \
    --type "claude" --vis

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_seg_multi.py \
    --model_path $model_name \
    --image_dir $data_path \
    --type "gemini" --vis

torchrun --standalone --nproc_per_node $gpus eval/evaluate_egoaff_seg_multi.py \
    --model_path $model_name \
    --image_dir $data_path \
    --type "gt" --vis