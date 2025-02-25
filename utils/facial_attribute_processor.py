import os
import json
import argparse
from tqdm import tqdm


def map_gender(age, gender):
    """Map gender based on age ranges."""
    if age <= 14:
        return "baby-boy" if gender == "man" else "baby-girl"
    elif 15 <= age <= 24:
        return "boy" if gender == "man" else "girl"
    else:
        return "man" if gender == "man" else "woman"


def process_images(image_folder, json_file, output_file):
    """Process images and generate sorted JSONL output."""
    # Load JSON metadata
    with open(json_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Collect valid entries
    entries = []

    # Get list of image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # Process images with tqdm for progress visualization
    for image_file in tqdm(image_files, desc="Processing images"):
        name, ext = os.path.splitext(image_file)
        if name in metadata:
            age = metadata[name]["age"]
            gender = metadata[name]["gender"]
            race = metadata[name]["race"]
            mapped_gender = map_gender(age, gender)
            prompt = f"{age}-year-old {race} {mapped_gender}"
            entries.append({"image": image_file, "prompt": prompt})

    # Sort entries by filename
    entries.sort(key=lambda x: x["image"])

    # Write to output JSONL file
    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry in entries:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sorted JSONL file with image descriptions."
    )
    parser.add_argument(
        "--image_folder", type=str, help="Path to the folder containing images"
    )
    parser.add_argument(
        "--json_file", type=str, help="Path to the input JSON metadata file"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save the sorted output JSONL file"
    )

    args = parser.parse_args()
    process_images(args.image_folder, args.json_file, args.output_file)

