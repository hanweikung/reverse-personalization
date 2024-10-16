export DATASET_LOADING_SCRIPT_PATH="test_dataset_loading_script.py"
export OUTPUT_DIR="test-infer"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG="INFO"

# CUDA_VISIBLE_DEVICES=0 python -m main_run_sdxl_batch \
accelerate launch --multi_gpu -m main_run_sdxl_batch \
	--dataset_loading_script_path=$DATASET_LOADING_SCRIPT_PATH --center_crop \
	--output_dir=$OUTPUT_DIR \
	--max_test_samples "1000" \
	--guidance_scale "5.0" \
	--skip "0.7" \
	--ip_adapter_scale "1.0" \
	--id_emb_scale "-1.0" \
	--resolution "1024" \
	--det_size "640" \
	--vis_input
