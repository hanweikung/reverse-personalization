import os
import json
import argparse
from tqdm import tqdm


def process_images(folder_path, json_file, output_file):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Get a sorted list of image files in the folder
    image_files = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Prepare to write to the output file
    with open(output_file, "w") as out_file:
        for image_file in tqdm(image_files, desc="Processing images"):
            # Get the filename without extension
            filename_without_ext = os.path.splitext(image_file)[0]

            # Check if the filename exists in the JSON data
            if filename_without_ext in data:
                attributes = data[filename_without_ext]
                prompt = f"{attributes['age']}-year-old {attributes['race']} {attributes['gender']}"
                entry = {"image": image_file, "prompt": prompt}

                # Write the entry to the output file
                out_file.write(json.dumps(entry) + "\n")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process images and generate JSON lines."
    )
    parser.add_argument("--folder", help="Path to the folder containing image files.")
    parser.add_argument("--json_file", help="Path to the JSON file with attributes.")
    parser.add_argument("--output_file", help="Path to the output JSON lines file.")

    args = parser.parse_args()

    # Call the processing function
    process_images(args.folder, args.json_file, args.output_file)


if __name__ == "__main__":
    main()
