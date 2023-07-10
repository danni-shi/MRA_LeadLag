source venv_leadlag/bin/activate
cd src

python3 clustering.py \
    --start 1005 \
    --end 5145 \
    --save-path '../results/real/2023-07-10-16h03min_clustering_full' \
    --use-save-path \
    --parallelize \
    
# Explanation on Boolean arguments: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse