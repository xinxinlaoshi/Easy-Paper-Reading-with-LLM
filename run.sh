#!/bin/bash

# Activate the conda virtual environment
source ~/anaconda3/etc/profile.d/conda.sh  # change to your own conda path
conda activate EasyPaperReadingEnv

python rag.py --directory data --index_name your_index_name --query_doc query.txt --answer_doc answer.txt

conda deactivate