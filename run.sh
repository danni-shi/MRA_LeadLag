source venv_leadlag/bin/activate
cd src

python3 clustering.py \
    --start 5 \
    --end 1000 \
    --save-path '../results/real/2023-07-11-11h34min_non-negative_affinity' \
    --use-save-path \
    --parallelize \
    
# Explanation on Boolean arguments: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse