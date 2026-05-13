#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

root_dir="$1"

if [[ ! -d "$root_dir" ]]; then
  echo "Error: '$root_dir' is not a directory."
  exit 1
fi

while IFS= read -r -d '' pdf_file; do
  svg_file="${pdf_file%.pdf}.svg"

  # Ensure existing output is replaced.
  rm -f "$svg_file"

  echo "Converting: $pdf_file -> $svg_file"
  pdftocairo -svg "$pdf_file" "$svg_file"
done < <(find "$root_dir" -type f -name '*.pdf' -print0)

echo "Done."