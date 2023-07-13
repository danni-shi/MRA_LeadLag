source venv_leadlag/bin/activate
cd src

# python3 main.py \
#     --start 5 \
#     --end 5146 \
#     --save-path '../results/real/full_non-negative_affinity' \
#     --real-data \
#     --parallelize

python3 trading_real.py
python3 visualization_trading.py