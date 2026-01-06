#!/bin/bash


script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Just in case

cd "$script_dir"

# Check if python3 or python is available and set `py_bin`
if command -v python3 >/dev/null 2>&1; then
	py_bin=python3
elif command -v python >/dev/null 2>&1; then
	py_bin=python
else
	echo "No python interpreter found (python3 or python)." >&2
	exit 1
fi

# Create a virtualenv in .venv if it doesn't already exist
if [ ! -d ".venv" ]; then
	echo "Creating virtualenv using $py_bin..."
	"$py_bin" -m venv .venv
fi

chmod +x ./__inenv

mkdir -p output_file
mkdir -p agent_scratchpad

# Philosophy: "During setup, we maintain for the first time"
# Helps DRY (Don't Repeat Yourself)
bash "$script_dir/maintenance_setup.sh"