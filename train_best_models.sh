#!/bin/bash

source .env/bin/activate

# 1o - Update database
python . -download-data -interval-list=1m,5m,15m,30m,1h -log-level=DEBUG -verbose

# 2o Prepare Best Params based on results of previous training
python . -prepare-best-parameters

# 3o Train Best Models with all data for input on Robo Trader
python . -batch-training -train-best-model #-update-data-from-web