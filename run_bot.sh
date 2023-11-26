#!/bin/bash

eval "$(conda shell.bash hook)"

cd /home/marcelo/des/mg_crypto_trader/
conda activate mg 
python /home/marcelo/des/mg_crypto_trader/. -run-bot -start-date=2023-01-01 -prediction-mode=index
