import argparse
import json
import os
from tqdm import tqdm


def replace_race(file_path, new_race, output_file):
    # Ensure the parent directory of the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with (
        open(file_path, "r", encoding="utf-8") as f,
        open(output_file, "w", encoding="utf-8") as out_f,
    ):
        lines = f.readlines()

        for line in tqdm(lines, desc="Processing entries"):
            try:
                data = json.loads(line.strip())

                # Replace the race in the prompt
                prompt_parts = data.get("prompt", "").split()
                for i, word in enumerate(prompt_parts):
                    if word in {
                        "asian",
                        "indian",
                        "black",
                        "white",
                        "middle-eastern",
                        "latino-hispanic",
                    }:
                        prompt_parts[i] = new_race
                        break

                data["prompt"] = " ".join(prompt_parts)
                out_f.write(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace race in a JSON lines file and save to a new file."
    )
    parser.add_argument("--file", help="Path to the input JSON lines file")
    parser.add_argument("--race", help="New race to replace in prompts")
    parser.add_argument("--output", help="Path to the output JSON lines file")
    args = parser.parse_args()

    replace_race(args.file, args.race, args.output)
