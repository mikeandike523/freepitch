#!/bin/bash

# Ensure requirements are installed

script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"


bash "$script_dir/__inenv" pip install -r "$script_dir/requirements.txt"