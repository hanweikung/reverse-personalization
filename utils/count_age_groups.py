import argparse
import json
import re
from collections import Counter


def categorize_age(age):
    if 0 <= age <= 14:
        return "Children (0-14 years)"
    elif 15 <= age <= 24:
        return "Youth (15-24 years)"
    elif 25 <= age <= 44:
        return "Young adults (25-44 years)"
    elif 45 <= age <= 64:
        return "Middle age adults (45-64 years)"
    else:
        return "Seniors (65+ years)"


def count_age_groups(file_path):
    age_counter = Counter()
    total_entries = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get("prompt", "")

                # Extract age from prompt using regex
                match = re.search(r"(\d+)-year-old", prompt)
                if match:
                    age = int(match.group(1))
                    category = categorize_age(age)
                    age_counter[category] += 1
                    total_entries += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    # Print age group counts and percentages
    for category, count in age_counter.items():
        percentage = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"{category}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count age groups in a JSON lines file."
    )
    parser.add_argument("file", help="Path to the JSON lines file")
    args = parser.parse_args()
    count_age_groups(args.file)

