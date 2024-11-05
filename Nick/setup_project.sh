#!/bin/bash

# -----------------------------------------
# Project Setup Script
# Description: This script creates the directory structure and 
# initial files for a project focused on analyzing LLM outputs 
# and detecting hallucinations.
# Created by: nicksnlp
# Source: https://chatgpt.com/share/672a97ca-f350-800b-b72e-b04f6c40c1eb
# -----------------------------------------

# Create the root project directory
mkdir -p project-root

# Create the data subdirectories
mkdir -p project-root/data/raw project-root/data/generated project-root/data/annotated project-root/data/logs

# Create the src subdirectory and scripts
mkdir -p project-root/src
touch project-root/src/__init__.py project-root/src/data_augmentation.py project-root/src/classifier.py project-root/src/combine_outputs.py project-root/src/retrieve_info.py

# Create directories for each team member's work
mkdir -p project-root/Sofia project-root/Zang project-root/Nick project-root/Xinyuan

# Create example scripts for each member
touch project-root/Sofia/develop_prompts.py project-root/Sofia/README.md
touch project-root/Zang/develop_prompts.py project-root/Zang/README.md
touch project-root/Nick/develop_prompts.py project-root/Nick/augment_data.py project-root/Nick/combine_outputs.py project-root/Nick/README.md
touch project-root/Xinyuan/train_classifier.py project-root/Xinyuan/README.md

# Create root-level files
touch project-root/.gitignore project-root/requirements.txt project-root/README.md

echo "Project structure created successfully!"

