source venv_leadlag/bin/activate
cd src

python3 clustering.py \
<<<<<<< HEAD
    --start 5145 \
    --end 5146 \
    --save-path '../results/real/2023-07-07-10h21min_non-negative_affinity' \
=======
    --start 5 \
    --end 1000 \
    --save-path '../results/real/2023-07-11-11h34min_non-negative_affinity' \
>>>>>>> b20f8fe9a6333631d6f34442baaeec7394d8ad80
    --use-save-path \
    --no-parallelize \
    
# Explanation on Boolean arguments: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse