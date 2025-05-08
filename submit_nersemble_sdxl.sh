#!/bin/bash

#SBATCH --job-name nersemble
#SBATCH --chdir .
#SBATCH --output O-%x.%j
#SBATCH --error E-%x.%j
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 0-00:30:00           # maximum execution time (HH:MM:SS)
#SBATCH --partition boost_usr_prod
#SBATCH --account IscrC_SYNFACE
#SBATCH --qos boost_qos_dbg

# Make conda available:
export PATH="/leonardo_scratch/fast/IscrC_SYNFACE/hkung/miniforge3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Activate conda environment:
conda activate diffusers-v0-30-0

export SCRIPT=main_run_sdxl.py

# Define face images for different runs
declare -A FACE_IMAGES_MAP
FACE_IMAGES_MAP["set1"]=" \
 /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/037/37_355.png \
 /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/037/37_414.png \
 /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/037/37_532.png \
"
FACE_IMAGES_MAP["set2"]=" \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/038/38_145.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/038/38_193.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/038/38_697.png \
"
FACE_IMAGES_MAP["set3"]=" \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/042/42_037.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/042/42_208.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/042/42_262.png \
"
FACE_IMAGES_MAP["set4"]=" \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/099/99_229.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/099/99_286.png \
  /leonardo_work/IscrC_SYNFACE/hkung/experiments3/datasets/nersemble/data/selected/whole_face_no_align/099/99_457.png \
"

# Map set keys to their respective negative prompts
declare -A NEGATIVE_PROMPTS
NEGATIVE_PROMPTS["set1"]="28-year-old indian man"
NEGATIVE_PROMPTS["set2"]="23-year-old asian girl"
NEGATIVE_PROMPTS["set3"]="30-year-old asian woman"
NEGATIVE_PROMPTS["set4"]="28-year-old middle eastern man"

run_script() {
  local input_image=$1
  local face_images=$2
  local negative_prompt=$3

  python $SCRIPT \
    --sd_model_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --input_image $input_image \
    --guidance_scale "-4.0" \
    --skip "0.7" \
    --ip_adapter_scale "0.4" \
    --id_emb_scale "1.0" \
    --inversion "leditspp" \
    --num_inversion_steps "100" \
    --resolution "1024" \
    --det_size "512" \
    --seed "0" \
    --face_images $face_images
  # --negative_prompt "$negative_prompt" # remove the `&` at the end so that it waits for each process to finish before moving on
}

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Loop over all sets and run the script
for set_key in "${!FACE_IMAGES_MAP[@]}"; do
  face_images="${FACE_IMAGES_MAP[$set_key]}"
  negative_prompt="${NEGATIVE_PROMPTS[$set_key]}"
  for image in $face_images; do
    run_script "$image" "$face_images" "$negative_prompt"
  done
done

wait
