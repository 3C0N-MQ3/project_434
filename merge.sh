#!/bin/bash

echo "---------- Merge Start ----------"
# The input file containing the indicators
input_file="main.py"
echo "Input file: $input_file"

# The resulting output file
output_file="merged.py"
echo "Output file: $output_file"

# Prepare an empty output file
: > "$output_file"

# Function to handle python tag replacement
handle_python_tag() {
  local file_path=$1
  echo "✓ - Python file found: $file_path"
  if [ -f "$file_path" ]; then
    cat "$file_path" >> "$output_file"
    echo "" >> "$output_file"  # Add a new line after the content
  else
    echo "x - Python file not found: $file_path"
  fi
}

# Read the input file line by line
while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ $line =~ \<py\!--\*\!\{(.*)\}--\> ]]; then
    file_path="${BASH_REMATCH[1]}"
    # Remove previous line if it contains """
    if [[ $(tail -n 1 "$output_file") == '"""' ]]; then
      sed -i '$d' "$output_file"
    fi
    handle_python_tag "$file_path"
    # Skip the next line if it contains """
    read -r next_line
    if [[ $next_line != '"""' ]]; then
      echo "$next_line" >> "$output_file"
    fi
  elif [[ $line =~ \<\!--\*\!\{(.*)\}--\> ]]; then
    file_path="${BASH_REMATCH[1]}"
    echo "✓ - File found: $file_path"
    if [ -f "$file_path" ]; then
      cat "$file_path" >> "$output_file"
    else
      echo "x - File not found: $file_path"
    fi
  else
    echo "$line" >> "$output_file"
  fi
done < "$input_file"

# Generating the jupyter notebook from the merged file
jupytext --set-formats py:percent,ipynb "$output_file"
echo "✓ - Jupyter notebook generated"

echo "---------- Merge Complete ----------"
