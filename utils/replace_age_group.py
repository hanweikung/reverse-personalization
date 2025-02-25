import argparse
import json
import random
import re
import os
from tqdm import tqdm


def get_age_range(age_group):
    age_ranges = {
        "children": (0, 14),
        "youth": (15, 24),
        "young adults": (25, 44),
        "middle age adults": (45, 64),
        "seniors": (65, 100),  # Assuming max age as 100
    }
    return age_ranges.get(age_group.lower(), None)


def update_gender_based_on_age(age, original_gender):
    if age <= 14:
        return (
            "baby-boy" if original_gender in ["man", "boy", "baby-boy"] else "baby-girl"
        )
    elif age <= 24:
        return "boy" if original_gender in ["man", "boy", "baby-boy"] else "girl"
    else:
        return "man" if original_gender in ["man", "boy", "baby-boy"] else "woman"


def replace_age_group(file_path, age_group, output_file):
    age_range = get_age_range(age_group)
    if not age_range:
        print(f"Invalid age group: {age_group}")
        print(
            "Available age groups: children, youth, young adults, middle age adults, seniors"
        )
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with (
        open(file_path, "r", encoding="utf-8") as f,
        open(output_file, "w", encoding="utf-8") as out_f,
    ):
        lines = f.readlines()

        for line in tqdm(lines, desc="Processing entries"):
            try:
                data = json.loads(line.strip())
                prompt = data.get("prompt", "")

                # Extract age and gender using regex
                age_match = re.search(r"(\d+)-year-old", prompt)
                gender_match = re.search(
                    r"\b(man|woman|boy|girl|baby-boy|baby-girl)\b", prompt
                )

                if not age_match or not gender_match:
                    print(f"Skipping invalid prompt format: {prompt}")
                    continue

                original_gender = gender_match.group(0)

                # Sample a random age from the specified age group
                new_age = random.randint(*age_range)
                new_gender = update_gender_based_on_age(new_age, original_gender)

                # Replace age and gender in the prompt
                prompt = re.sub(r"\d+-year-old", f"{new_age}-year-old", prompt)
                prompt = re.sub(
                    r"\b(man|woman|boy|girl|baby-boy|baby-girl)\b", new_gender, prompt
                )

                data["prompt"] = prompt
                out_f.write(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")


if __name__ == "__main__":
    age_groups = "children, youth, young adults, middle age adults, seniors"
    parser = argparse.ArgumentParser(
        description=f"Replace age group in a JSON lines file and save to a new file. Available age groups: {age_groups}"
    )
    parser.add_argument("--file", help="Path to the input JSON lines file")
    parser.add_argument(
        "--age_group",
        help=f"New age group to replace in prompts (case insensitive). Options: {age_groups}",
    )
    parser.add_argument("--output", help="Path to the output JSON lines file")
    args = parser.parse_args()

    replace_age_group(args.file, args.age_group, args.output)
