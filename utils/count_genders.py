import json
import argparse
from tqdm import tqdm


def count_genders(json_lines):
    # Define the terms for males and females
    male_terms = ["man", "boy", "baby-boy"]
    female_terms = ["woman", "girl", "baby-girl"]

    # Initialize counters
    male_count = 0
    female_count = 0

    for line in tqdm(json_lines):
        entry = json.loads(line)
        prompt = entry["prompt"]

        # Split the prompt on whitespace
        words = prompt.split()

        # Count based on whole words
        for word in words:
            if word in male_terms:
                male_count += 1
            elif word in female_terms:
                female_count += 1

    total_count = male_count + female_count
    male_percentage = (male_count / total_count * 100) if total_count > 0 else 0
    female_percentage = (female_count / total_count * 100) if total_count > 0 else 0

    return male_count, female_count, male_percentage, female_percentage


def main(input_file):
    with open(input_file, "r") as infile:
        json_lines = infile.readlines()

    # Count genders and print results
    male_count, female_count, male_percentage, female_percentage = count_genders(
        json_lines
    )

    print(f"Number of men: {male_count} ({male_percentage:.2f}%)")
    print(f"Number of women: {female_count} ({female_percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count gender in JSON lines file.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSON lines file"
    )

    args = parser.parse_args()

    main(args.input_file)
