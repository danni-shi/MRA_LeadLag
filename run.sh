source venv_leadlag/bin/activate
cd src

python3 clustering.py \
    --start 5145 \
    --end 5146 \
    --save-path '../results/real/2023-07-07-10h21min_non-negative_affinity' \
    --use-save-path \
    --no-parallelize \
    
# Explanation on Boolean arguments: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse