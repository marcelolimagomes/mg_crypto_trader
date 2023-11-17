#!/bin/bash

conda activate mg

# python . -batch-training -calc-rsi -verbose -prediction-mode=index -symbol-list=BTCUSDT -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,150 -target-margin-list=1.0
# python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,150 -target-margin-list=1.0

#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1h -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=50,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 -log-level=WARNING
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1h -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=100,150 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1h -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,200 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1h -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=200,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0

python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=30m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=50,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 -log-level=WARNING
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=30m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=100,150 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=30m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,200 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=30m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=200,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0

python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=15m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=50,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 -log-level=WARNING
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=15m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=100,150 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=15m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,200 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=15m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=200,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0

python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=5m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=50,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 -log-level=WARNING
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=5m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=100,150 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=5m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,200 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=5m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=200,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0

python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=50,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 -log-level=WARNING
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=100,150 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=150,200 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#python . -batch-training -calc-rsi -verbose -prediction-mode=index -interval-list=1m -range-min-rsi=28 -range-max-rsi=80 -range-p-ema=200,250 -target-margin-list=1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
