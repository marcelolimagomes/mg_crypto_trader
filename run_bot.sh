#!/bin/bash

eval "$(conda shell.bash hook)"

cd /home/marcelo/des/mg_crypto_trader/
conda activate mg 
python /home/marcelo/des/mg_crypto_trader/. -download-data -interval-list=1h,30m,15m,5m,1m -verbose
python /home/marcelo/des/mg_crypto_trader/. -run-bot -prediction-mode=index
