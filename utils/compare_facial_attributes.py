import json
import argparse
import os
from tqdm import tqdm


def load_json(file_path):
    """Loads a JSON file and returns the data as a dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_dicts(file1, file2, age_range):
    """Compares two dictionaries and calculates match percentages for race, gender, and age."""
    keys = set(file1.keys()) & set(file2.keys())  # Only compare matching keys

    total = len(keys)
    if total == 0:
        return 0, 0, 0  # Avoid division by zero

    race_matches = 0
    gender_matches = 0
    age_matches = 0

    for key in tqdm(keys, desc="Processing records"):
        if file1[key]["race"] == file2[key]["race"]:
            race_matches += 1
        if file1[key]["gender"] == file2[key]["gender"]:
            gender_matches += 1
        if abs(file1[key]["age"] - file2[key]["age"]) <= age_range:
            age_matches += 1

    race_percentage = round((race_matches / total) * 100, 3)
    gender_percentage = round((gender_matches / total) * 100, 3)
    age_percentage = round((age_matches / total) * 100, 3)

    return race_percentage, gender_percentage, age_percentage


def save_results(output_path, race, gender, age):
    """Saves the computed percentages to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Race Match Percentage: {race}%\n")
        f.write(f"Gender Match Percentage: {gender}%\n")
        f.write(f"Age Close Percentage: {age}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSON files containing demographic data."
    )
    parser.add_argument("--file1", help="Path to the first JSON file")
    parser.add_argument("--file2", help="Path to the second JSON file")
    parser.add_argument(
        "--age_range",
        type=int,
        default=2,
        help="Allowed age difference range for a match",
    )
    parser.add_argument("--output", required=True, help="Path for the output text file")

    args = parser.parse_args()

    data1 = load_json(args.file1)
    data2 = load_json(args.file2)

    race, gender, age = compare_dicts(data1, data2, args.age_range)
    save_results(args.output, race, gender, age)

    print(f"Results saved to {args.output}")
    print(f"Race Match: {race}%\nGender Match: {gender}%\nAge Close: {age}%")


if __name__ == "__main__":
    main()
