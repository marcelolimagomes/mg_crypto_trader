#!/bin/bash
:'
python . -download-data -interval-list=1h
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=1h -estimator-list=knn -stop-loss-list=1.5 \
-numeric-features=volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=24 -regression-times-list=0 -start-train-date=2017-11-01 -start-test-date=2023-07-01


python . -download-data -interval-list=30m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=30m -estimator-list=knn -stop-loss-list=1.5 \
-numeric-features=volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=48 -regression-times-list=0 -start-train-date=2020-10-01 -start-test-date=2023-07-01

python . -download-data -interval-list=15m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=15m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=60 -regression-times-list=0 -start-train-date=2022-02-01 -start-test-date=2023-07-01

python . -download-data -interval-list=5m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=5m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=72 -regression-times-list=0 -start-train-date=2023-01-01 -start-test-date=2023-07-01

python . -download-data -interval-list=1m 
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=1m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=120 -regression-times-list=0 -start-train-date=2023-06-01 -start-test-date=2023-07-01
'

## New Features
python . -download-data -interval-list=1h
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=1h -estimator-list=knn -stop-loss-list=1.5 \
-numeric-features=close,volume,quote_asset_volume,number_of_trades \
-regression-PnL-list=24 -regression-times-list=0 -start-train-date=2017-11-01 -start-test-date=2023-07-01

:'
python . -download-data -interval-list=30m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=30m -estimator-list=knn -stop-loss-list=1.5 \
-numeric-features=close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume \
-regression-PnL-list=48 -regression-times-list=0 -start-train-date=2020-10-01 -start-test-date=2023-07-01
'

python . -download-data -interval-list=15m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=15m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=close,volume,quote_asset_volume,number_of_trades \
-regression-PnL-list=60 -regression-times-list=0 -start-train-date=2022-02-01 -start-test-date=2023-07-01

python . -download-data -interval-list=5m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=5m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=close,volume,quote_asset_volume,number_of_trades \
-regression-PnL-list=72 -regression-times-list=0 -start-train-date=2023-01-01 -start-test-date=2023-07-01

python . -download-data -interval-list=1m
python . -batch-training -calc-rsi -normalize -no-tune -n-jobs=20 \
-symbol-list=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SOLUSDT,TRXUSDT,DOTUSDT,MATICUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,XLMUSDT,LINKUSDT,XMRUSDT,UNIUSDT,ATOMUSDT \
-interval-list=1m -estimator-list=knn -stop-loss-list=1.0 \
-numeric-features=close,volume,quote_asset_volume,number_of_trades \
-regression-PnL-list=120 -regression-times-list=0 -start-train-date=2023-06-01 -start-test-date=2023-07-01
