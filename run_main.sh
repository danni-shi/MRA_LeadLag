source venv_leadlag/bin/activate
cd src

python3 main.py \
    --start 45 \
    --end 105 \
    --save-path '../results/real/full_non-negative_affinity' \
    --real-data \
    --parallelize