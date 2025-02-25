import argparse
import json
from collections import Counter


def count_race_groups(file_path):
    race_categories = {
        "asian",
        "indian",
        "black",
        "white",
        "middle-eastern",
        "latino-hispanic",
    }
    race_counter = Counter()
    total_entries = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get("prompt", "").lower()

                # Find the race mentioned in the prompt
                for race in race_categories:
                    if race in prompt:
                        race_counter[race] += 1
                        total_entries += 1
                        break
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    # Print race counts and percentages
    for race, count in race_counter.items():
        percentage = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"{race}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count race groups in a JSON lines file."
    )
    parser.add_argument("file", help="Path to the JSON lines file")
    args = parser.parse_args()
    count_race_groups(args.file)
