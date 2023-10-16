# %%
import os
import sys
sys.path.insert(0, sys.path[0].removesuffix('/src/jupyter_nb'))
print(sys.path[0])
import pandas as pd
from pycaret.classification import ClassificationExperiment
import src.utils as utils
import src.calcEMA as calc_utils
import src.myenv as myenv
from datetime import datetime

from itertools import combinations
import plotly.express as px

# %%
# Variables
# ETCUSDT BTCUSDT
# symbol = 'ETHUSDT'
symbol = 'BTCUSDT'
interval = '1m'
# lightgbm  xgboost lr et rf
estimator = 'knn'
_compare_models = False

start_train_date = '2022-01-01'  # train < and test >=
start_test_date = '2023-01-01'  # train < and test >=

stop_loss = 1.0
# regression_times = 0  # 24 * 30 * 2  # horas
times_regression_PnL = 120
normalize = True
use_gpu = True
tune_model = False
apply_combination_features = False

# %% [markdown]
# ### Metadata
#
# <code>
# Field Name - Description</br>
# open_time - Kline Open time in unix time format</br>
# open - Open Price</br>
# high - High Price</br>
# low	- Low Price</br>
# close	- Close Price</br>
# volume - Volume</br>
# close_time - Kline Close time in unix time format</br>
# quote_volume - Quote Asset Volume</br>
# count	- Number of Trades</br>
# taker_buy_volume - Taker buy base asset volume during this period</br>
# taker_buy_quote_volume - Taker buy quote asset volume during this period</br>
# ignore - Ignore</br>
# </code>

# %%
cols = myenv.all_klines_cols.copy()
cols.remove('ignore')
data = utils.get_data(symbol=symbol, save_database=False, interval=interval, tail=-1, columns=cols, parse_dates=True)
data = data[data['open_time'] >= start_train_date]
data = utils.parse_type_fields(data, parse_dates=True)
data = utils.adjust_index(data)
data.info()
data

# %%
data = calc_utils.calc_RSI(data)
data.info()
data

# %%
data = calc_utils.calc_ema_periods(data, periods_of_time=[times_regression_PnL, 200])
data.info()
data

# %%
data = utils.regression_PnL(
    data=data,
    label=myenv.label,
    diff_percent=float(stop_loss),
    max_regression_profit_and_loss=int(times_regression_PnL),
    drop_na=True,
    drop_calc_cols=True,
    strategy=None)
data.info()
data

# %%
perc_data_label = data[[myenv.label, 'open_time']].groupby(myenv.label).count()
perc_data_label['perc'] = perc_data_label['open_time'] / len(data)
perc_data_label

# %%
train_data = data[(data['open_time'] >= start_train_date) & (data['open_time'] < start_test_date)]
train_data = train_data.sort_values(myenv.date_features)
train_data

# %%
test_data = data[data['open_time'] >= start_test_date]
test_data = test_data.sort_values(myenv.date_features)
test_data

# %%
# BTCUSDT 1h best params: close,volume,quote_asset_volume,number_of_trades,rsi
# numeric_features = 'volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,rsi,ema_24p,ema_200p'.split(',')
# text_numeric_features = 'close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,rsi,ema_24p,ema_200p'
text_numeric_features = f'number_of_trades,rsi,ema_{int(times_regression_PnL)}p,ema_200p'
numeric_features = text_numeric_features.split(',')
print(f'Numeric Features: {numeric_features} - size: {len(numeric_features)}\n')

combination_numeric_features = []
if apply_combination_features:
  for size in range(1, len(numeric_features) + 1):
    comb = map(list, combinations(numeric_features, size))
    for c in comb:
      res = ''
      for j in c:
        res += f'{j},'
      combination_numeric_features.append(res[0:len(res) - 1])
else:
  combination_numeric_features = [text_numeric_features]

print(f'Combination Numeric Features: \n{combination_numeric_features}')

# %%
from datetime import datetime
# fix_imbalance_method: condensednearestneighbour, editednearestneighborus, repeatededitednearestneighbours, allknn, instancehardnessthreshold, nearmiss, neighbourhoodcleaningrule, onesidedselection, randomundersampler, tomeklinks, randomoversampler, smote, smotenc, smoten, adasyn, borderlinesmote, kmeanssmote, svmsmote, smoteenn, smotetomek.
# 'condensednearestneighbour,editednearestneighborus,repeatededitednearestneighbours,allknn,instancehardnessthreshold,nearmiss,neighbourhoodcleaningrule,onesidedselection,randomundersampler,tomeklinks,randomoversampler,smote,smotenc,smoten,adasyn,borderlinesmote,kmeanssmote,svmsmote,smoteenn,smotetomek'.split(',')
# 'smotenc,smoten,adasyn,borderlinesmote,kmeanssmote,svmsmote,smoteenn,smotetomek'.split(',')
# imbalance_list = 'editednearestneighborus,repeatededitednearestneighbours,allknn,instancehardnessthreshold,nearmiss,neighbourhoodcleaningrule,onesidedselection,randomundersampler,tomeklinks,randomoversampler,smote,smotenc,smoten,adasyn,borderlinesmote,kmeanssmote,svmsmote,smoteenn,smotetomek'.split(',')
imbalance_list = 'instancehardnessthreshold,smoteenn,repeatededitednearestneighbours,allknn'.split(',')
simulation_results_filename = f'./resultado_simulacao_{symbol}_{interval}.csv'

df_resultado_simulacao = pd.DataFrame()
has_simulation_file = os.path.exists(simulation_results_filename)
if has_simulation_file:
  df_resultado_simulacao = pd.read_csv(simulation_results_filename, sep=';')
for aux_numeric_features in combination_numeric_features:
  experiement = ClassificationExperiment()
  features = []
  features += ['open_time', myenv.label]
  features += aux_numeric_features.split(',')
  print(f'features: {features}')
  for fix_imbalance_method in imbalance_list:
    if has_simulation_file:
      chave = (df_resultado_simulacao['symbol'] == symbol) & \
          (df_resultado_simulacao['interval'] == interval) & \
          (df_resultado_simulacao['estimator'] == estimator) & \
          (df_resultado_simulacao['fix_imbalance_method'] == fix_imbalance_method) & \
          (df_resultado_simulacao['features'] == str(features))

      if chave.sum() > 0:
        print(f'fix_imbalance_method: {fix_imbalance_method} already exists')
        continue

    try:
      print(f'fix_imbalance_method: {fix_imbalance_method}')
      setup = experiement.setup(
          data=train_data[features].copy(),
          train_size=myenv.train_size,
          target=myenv.label,
          numeric_features=aux_numeric_features.split(','),
          date_features=['open_time'],
          create_date_columns=["hour", "day", "month"],
          data_split_shuffle=False,
          data_split_stratify=False,
          fix_imbalance=True,
          fix_imbalance_method=fix_imbalance_method,
          remove_outliers=True,
          fold_strategy='timeseries',
          fold=3,
          session_id=123,
          normalize=normalize,
          use_gpu=use_gpu,
          verbose=True,
          n_jobs=20,
          log_experiment=False,
      )

      if _compare_models:
        best = setup.compare_models()
        estimator = setup.pull().index[0]
        print('Estimator: ' + estimator)
      else:
        best = setup.create_model(estimator)

      if tune_model:
        best = setup.tune_model(best)

      # predict on test set
      # holdout_pred = setup.predict_model(best)
      # print('Holdout Score:', holdout_pred['prediction_score'].mean())
      # print('Holdout Score Group:\n', holdout_pred[[myenv.label, 'prediction_score']].groupby(myenv.label).mean())

      predict = setup.predict_model(best, data=test_data.drop(columns=[myenv.label]))
      predict[myenv.label] = test_data[myenv.label]
      predict['_score'] = predict['prediction_label'] == predict[myenv.label]
      # print('Predict Score Mean:', predict['_score'].mean())
      # print('Predict Score Mean Group:\n', predict[[myenv.label, '_score']].groupby(myenv.label).mean())

      final_model = setup.finalize_model(best)

      ajusted_test_data = test_data.drop(myenv.label, axis=1)
      df_final_predict, res_score = utils.validate_score_test_data(
          setup,
          final_model,
          myenv.label,
          test_data,
          ajusted_test_data)

      # df_final_predict.info()
      # print('Final Score Mean:', res_score.mean().values[0])
      # print('Final Score Group:\n', res_score)

      start_test_date = df_final_predict['open_time'].min()
      end_test_date = df_final_predict['open_time'].max()

      # print('Simule Trading:')
      # print(f'Min Data: {start_test_date}')
      # print(f'Max Data: {end_test_date}')
      saldo_inicial = 100.0
      saldo_final = utils.simule_trading_crypto2(df_final_predict, start_test_date, end_test_date, saldo_inicial, stop_loss)
      print(f'>>>> Saldo Final: {saldo_final} - features: {features}\n\n')

      result_simulado = {}
      result_simulado['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      result_simulado['symbol'] = symbol
      result_simulado['interval'] = interval
      result_simulado['estimator'] = estimator
      result_simulado['fix_imbalance_method'] = fix_imbalance_method
      result_simulado['stop_loss'] = stop_loss
      result_simulado['times_regression_profit_and_loss'] = times_regression_PnL
      result_simulado['features'] = features
      result_simulado['final_value'] = saldo_final

      if res_score is not None:
        result_simulado['score'] = ''
        for i in range(0, len(res_score.index.values)):
          result_simulado['score'] += f'[{res_score.index.values[i]}={res_score["_score"].values[i]:.2f}]'

      df_resultado_simulacao = pd.concat([df_resultado_simulacao, pd.DataFrame([result_simulado])], ignore_index=True)
      df_resultado_simulacao.sort_values('final_value', inplace=True)

      df_resultado_simulacao.to_csv(simulation_results_filename, sep=';', index=False)
    except Exception as e:
      print(e)
      continue
