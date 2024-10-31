export SD_MODEL_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_LOADING_SCRIPT_PATH="test_dataset_loading_script.py"
export OUTPUT_DIR="test-infer"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG="INFO"

# CUDA_VISIBLE_DEVICES=0 python -m main_run_batch \
accelerate launch --multi_gpu -m main_run_batch \
	--sd_model_path=$SD_MODEL_PATH \
	--dataset_loading_script_path=$DATASET_LOADING_SCRIPT_PATH \
	--center_crop \
	--output_dir $OUTPUT_DIR \
	--max_test_samples "1000" \
	--guidance_scale "7.0" \
	--skip "70" \
	--ip_adapter_scale "1.0" \
	--id_emb_scale "1.0" \
	--det_size "640" \
	--det_thresh "0.1" \
	--seed "0" \
	--max_angle "0.0" \
	--mask_delay_steps "0" \
	--vis_input
