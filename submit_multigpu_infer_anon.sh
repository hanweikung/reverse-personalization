#!/bin/bash

#SBATCH --job-name=celeba
#SBATCH --chdir=.
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH --cpus-per-task=8           # number of CPUs per task
#SBATCH --time=1-00:00:00           # maximum execution time (HH:MM:SS)
#SBATCH --partition gpupart
#SBATCH --account staff

######################
### Set enviroment ###
######################
# Make conda available:
export PATH="/nfs/data_todi/hkung/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Activate a conda environment:
conda activate diffusers-v0-30-0

export GPUS_PER_NODE=2
export MAIN_PROCESS_PORT=29500
######################

export SCRIPT=main_run_batch
export SCRIPT_ARGS=" \
  --sd_model_path=/path/to/stable-diffusion-v1-5/ \
  --insightface_model_path=/path/to/insightface/ \
  --dataset_loading_script_path=test_dataset_loading_script.py \
  --center_crop \
  --output_dir=/path/to/output/directory/ \
  --max_test_samples "1000" \
  --guidance_scale "10.0" \
  --skip "70" \
  --ip_adapter_scale "1.0" \
  --id_emb_scale "1.0" \
  --det_size "640" \
  --det_thresh "0.1" \
  --seed "0" \
  --max_angle "15.0" \
  --mask_delay_steps "30" \
  --vis_input
  "

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT # necessary for proper job cleanup

accelerate launch --num_processes $GPUS_PER_NODE --main_process_port $MAIN_PROCESS_PORT -m $SCRIPT $SCRIPT_ARGS & # & is necessary for proper job cleanup

wait # necessary for proper job cleanup
