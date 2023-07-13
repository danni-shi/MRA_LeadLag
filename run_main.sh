source venv_leadlag/bin/activate
cd src

python3 main.py \
     --num-rounds 8 \
     --no-real-data \
     --parallelize \
     --folder-name '8rounds'

#python3 trading_real.py
#python3 visualization_trading.py