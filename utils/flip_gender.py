import json
import argparse
from tqdm import tqdm


def flip_gender(json_lines):
    # Define a mapping for gender flipping
    gender_flip = {
        "baby-boy": "baby-girl",
        "baby-girl": "baby-boy",
        "man": "woman",
        "woman": "man",
        "boy": "girl",
        "girl": "boy",
    }

    flipped_entries = []

    # Use tqdm to visualize progress
    for line in tqdm(json_lines, desc="Processing entries"):
        entry = json.loads(line)
        prompt = entry["prompt"]

        # Split the prompt to identify the gender
        parts = prompt.split()

        # Check if the gender is in the prompt and flip it
        for i in range(len(parts)):
            if parts[i] in gender_flip:
                parts[i] = gender_flip[parts[i]]
                break  # Only flip the first occurrence

        # Reconstruct the prompt
        flipped_prompt = " ".join(parts)
        flipped_entry = {"image": entry["image"], "prompt": flipped_prompt}

        flipped_entries.append(json.dumps(flipped_entry))

    return flipped_entries


def main(input_file, output_file):
    with open(input_file, "r") as infile:
        json_lines = infile.readlines()

    flipped_results = flip_gender(json_lines)

    with open(output_file, "w") as outfile:
        for result in flipped_results:
            outfile.write(result + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip gender in JSON lines file.")
    parser.add_argument(
        "--input_file", type=str, help="Path to the input JSON lines file"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save the output JSON lines file"
    )

    args = parser.parse_args()

    main(args.input_file, args.output_file)
