from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.models import models
import src.utils as utils
import src.myenv as myenv
import src.send_message as sm
import pandas as pd
import numpy as np
import time
import src.calcEMA as calc_utils
import os
import logging


class RoboTrader():
  def __init__(self, params: dict):
    #print('ROBO TRADER >>>>>>> ', params)
    self._params = params
    self._all_data = {}
    self._all_data['CAI'] = params['CAI']['all_data'].copy() if 'CAI' in params else None
    self._all_data['SOBE'] = params['SOBE']['all_data'].copy() if 'SOBE' in params else None

    # Single arguments
    self._symbol = params['symbol']
    self._interval = params['interval']
    self._mutex = params['mutex']

    self._estimator = utils.get_params_robo_trader(params, 'estimator')  # params['estimator']
    self._imbalance_method = utils.get_params_robo_trader(params, 'imbalance_method') # params['imbalance_method']

    # List arguments
    self._start_date = utils.get_params_robo_trader(params, 'start_date')
    self._numeric_features = utils.get_params_robo_trader(params, 'numeric_features', split=True)

    self._stop_loss = utils.get_params_robo_trader(params, 'stop_loss', 'float')
    self._regression_times = utils.get_params_robo_trader(params, 'regression_times', 'int')

    if (self._regression_times['CAI'] > 0) or (self._regression_times['SOBE'] > 0):
      self._regression_features = utils.get_params_robo_trader(params, 'regression_features', split=True)
    else:
      self._regression_features = []

    self._times_regression_profit_and_loss = utils.get_params_robo_trader(params, 'times_regression_profit_and_loss', 'int')
    # Boolean arguments
    self._calc_rsi = utils.get_params_robo_trader(params, 'calc_rsi')
    self._arguments = utils.get_params_robo_trader(params, 'arguments')

    if 'CAI' in params:
      self._verbose = params['CAI']['verbose']  
    elif 'SOBE' in params:
      self._verbose =  params['SOBE']['verbose']
    else:
      self._verbose = False

    # Internal atributes
    self._features_added = {'CAI': [], 'SOBE': []}
    self._experiment = {'CAI': None, 'SOBE': None}
    self._setup = {'CAI': None, 'SOBE': None}
    self._model = {'CAI': None, 'SOBE': None}
    self._model_name_init = {'CAI': '', 'SOBE': ''}

    # Prepare columns to kline
    self._kline_features = {}    
    self._kline_features['CAI'] = myenv.date_features + self._numeric_features['CAI'] if 'CAI' in params else None
    self._kline_features['SOBE'] = myenv.date_features + self._numeric_features['SOBE'] if 'SOBE' in params else None
    
    if 'CAI' in params and 'close' not in self._kline_features['CAI']:
      self._kline_features['CAI'].append('close')
      
    if 'SOBE' in params and  'close' not in self._kline_features['SOBE']:
      self._kline_features['SOBE'].append('close')

    self._all_features = {'CAI': [], 'SOBE': []}

    # Initialize logging
    if 'CAI' in params:
      self.log = self._configure_log(params['CAI']['log_level'])
    elif 'SOBE' in params:
      self.log = self._configure_log(params['SOBE']['log_level'])
    else:
      self.log = self._configure_log(myenv.log_level)

    self.ix_symbol = f'{self._symbol}_{self._interval}'
    sm.send_status_to_telegram(f'Starting Robo Trader for {self._symbol}_{self._interval}')

  def _configure_log(self, log_level):
    log_file_path = os.path.join(myenv.logdir, f'robo_trader_{self._symbol}_{self._interval}.log')
    logger = logging.getLogger(f'robo_trader_{self._symbol}_{self._interval}')
    logger.propagate = False

    logger.setLevel(log_level)

    fh = logging.FileHandler(log_file_path, mode='a', delay=True)
    fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval} - %(message)s', '%Y-%m-%d %H:%M:%S'))
    fh.setLevel(log_level)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - RoboTrader: {self._symbol}-{self._interval} - %(message)s', '%Y-%m-%d %H:%M:%S'))
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

  def _data_preprocessing(self):
    return

  def _feature_engineering(self):
    if 'CAI' in self._all_features and self._kline_features['CAI'] is not None:
      self._all_features['CAI'] += self._kline_features['CAI']

    if 'SOBE' in self._all_features and self._kline_features['SOBE'] is not None:
      self._all_features['SOBE'] += self._kline_features['SOBE']

    if 'CAI' in self._all_features:
      self._all_features['CAI'] += ['rsi'] if self._calc_rsi['CAI'] and 'rsi' not in self._all_features['CAI'] else []
    if 'SOBE' in self._all_features:
      self._all_features['SOBE'] += ['rsi'] if self._calc_rsi['SOBE'] and 'rsi' not in self._all_features['SOBE'] else []

    if 'CAI' in self._regression_times and self._regression_times['CAI'] > 0:
      self.log.info(f'calculating regresstion_times: {self._regression_times["CAI"]} - regression_features: {self._regression_features["CAI"]}')
      self._all_data['CAI'], self._features_added['CAI'] = utils.regresstion_times(
          self._all_data['CAI'],
          self._regression_features['CAI'],
          self._regression_times['CAI'],
          last_one=False)
      self.log.info(f'Info after calculating regresstion_times["CAI"]: ') if self._verbose else None
      self._all_data['CAI'].info() if self._verbose else None
      self._all_features['CAI'] += self._features_added['CAI']
      self._all_data['CAI'] = self._all_data['CAI'][self._kline_features['CAI']] if 'CAI' in self._all_data else None
      self._all_data['CAI'].info() if 'CAI' in self._all_data else None
      self.log.info(f'All Features CAI: {self._all_features["CAI"]}')
  

    if 'SOBE' in self._regression_times and self._regression_times['SOBE'] > 0:
      self.log.info(f'calculating regresstion_times: {self._regression_times["SOBE"]} - regression_features: {self._regression_features["SOBE"]}')
      self._all_data['SOBE'], self._features_added['SOBE'] = utils.regresstion_times(
          self._all_data['SOBE'],
          self._regression_features['SOBE'],
          self._regression_times['SOBE'],
          last_one=False)
      self.log.info(f'Info after calculating regresstion_times["SOBE"]: ') if self._verbose else None
      self._all_data['SOBE'].info() if self._verbose else None
      self._all_features['SOBE'] += self._features_added['SOBE']
      self._all_data['SOBE'] = self._all_data['SOBE'][self._kline_features['SOBE']] if 'SOBE' in self._all_data else None
      self._all_data['SOBE'].info() if 'SOBE' in self._all_data else None
      self.log.info(f'All Features SOBE: {self._all_features["SOBE"]}') 

    self.log.debug('Reshape to All Features')  

  def _load_model(self):
    self._model_name_init['CAI'] = utils.get_model_name_to_load(
        'CAI',
        self._symbol,
        self._interval,
        self._estimator['CAI'],
        self._stop_loss['CAI'],
        self._regression_times['CAI'],
        self._times_regression_profit_and_loss['CAI']) if 'CAI' in self._model_name_init else ''

    self._model_name_init['SOBE'] = utils.get_model_name_to_load(
        'SOBE',
        self._symbol,
        self._interval,
        self._estimator['SOBE'],
        self._stop_loss['SOBE'],
        self._regression_times['SOBE'],
        self._times_regression_profit_and_loss['SOBE']) if 'SOBE' in self._model_name_init else ''

    self._experiment['CAI'], self._model['CAI'] = utils.load_model(
        'CAI',
        self._symbol,
        self._interval,
        self._estimator['CAI'],
        self._stop_loss['CAI'],
        self._regression_times['CAI'],
        self._times_regression_profit_and_loss['CAI']) if 'CAI' in self._model_name_init else None
    
    self._experiment['SOBE'], self._model['SOBE'] = utils.load_model(
        'SOBE',
        self._symbol,
        self._interval,
        self._estimator['SOBE'],
        self._stop_loss['SOBE'],
        self._regression_times['SOBE'],
        self._times_regression_profit_and_loss['SOBE']) if 'SOBE' in self._model_name_init else None

    self.log.info(f'model {self._model_name_init["CAI"]} loaded.')
    self.log.info(f'model {self._model_name_init["SOBE"]} loaded.')

  def update_data_from_web(self):
    #self._kline_features
    df_klines = utils.get_klines(symbol=self._symbol, interval=self._interval, max_date=None, limit=3, columns=myenv.all_klines_cols, parse_dates=True)
    # df_klines['symbol'] = self._symbol
    self.log.debug(f'df_klines shape: {df_klines.shape}')
    df_klines.info() if self._verbose else None

    if 'CAI' in self._all_data and self._all_data["CAI"] is not None:
      self.log.debug(f'Getting data from web. all_data shape CAI: {self._all_data["CAI"].shape}')
      klines_features = utils.remove_cols_for_klines(self._kline_features['CAI'])
      self.log.debug(f'klines_features CAI: {klines_features}')
      self._all_data['CAI'] = pd.concat([self._all_data['CAI'], df_klines[klines_features]])
      self._all_data['CAI'].drop_duplicates(keep='last', subset=['open_time'], inplace=True)
      self._all_data['CAI'].sort_index(inplace=True)
      self.log.debug(f'Updated - all_data.shape CAI: {self._all_data["CAI"].shape}')

    if 'SOBE' in self._all_data and self._all_data["SOBE"] is not None:
      self.log.debug(f'Getting data from web. all_data shape SOBE: {self._all_data["SOBE"].shape}')
      klines_features = utils.remove_cols_for_klines(self._kline_features['SOBE'])
      self.log.debug(f'klines_features SOBE: {klines_features}')
      self._all_data['SOBE'] = pd.concat([self._all_data['SOBE'], df_klines[klines_features]])
      self._all_data['SOBE'].drop_duplicates(keep='last', subset=['open_time'], inplace=True)
      self._all_data['SOBE'].sort_index(inplace=True)
      self.log.debug(f'Updated - all_data.shape SOBE: {self._all_data["SOBE"].shape}')

    now_price = df_klines.tail(1)['close'].values[0]
    self.log.info(f'{self._symbol} Price: {now_price:.6f}')

    latest_closed_candle_open_time = df_klines.iloc[df_klines.shape[0] - 2:df_klines.shape[0] - 1]['open_time'].values[0]
    return now_price, latest_closed_candle_open_time

  def feature_engineering_on_loop(self):
    self.log.info(f'Calculating EMA\'s for key {self._symbol}_{self._interval}...')

    rsi = {}
    if 'CAI' in self._all_data and self._all_data['CAI'] is not None:
      self._all_data['CAI'] = calc_utils.calc_ema_periods(self._all_data['CAI'], periods_of_time=[self._times_regression_profit_and_loss['CAI'], 200])
      if self._calc_rsi['CAI']:
        self.log.debug(f'Start Calculating RSI CAI ...')
        self._all_data['CAI'] = calc_utils.calc_RSI(self._all_data['CAI'])
        rsi['CAI'] = self._all_data['CAI'].tail(1)['rsi'].values[0]
        self.log.debug(f'After Calculating RSI. all_data.shape CAI: {self._all_data["CAI"].shape}')

      self.log.debug(f'regression_times CAI: {self._regression_times["CAI"]}')
      if (self._regression_times['CAI'] is not None) and (self._regression_times['CAI'] > 0):
        self._all_data['CAI'], _ = utils.regresstion_times(self._all_data['CAI'], self._numeric_features['CAI'], self._regression_times['CAI'], last_one=True)

    if 'SOBE' in self._all_data and self._all_data['SOBE'] is not None:
      self._all_data['SOBE'] = calc_utils.calc_ema_periods(self._all_data['SOBE'], periods_of_time=[self._times_regression_profit_and_loss['SOBE'], 200])
      if self._calc_rsi['SOBE']:
        self.log.debug(f'Start Calculating RSI SOBE...')
        self._all_data['SOBE'] = calc_utils.calc_RSI(self._all_data['SOBE'])
        rsi['SOBE'] = self._all_data['SOBE'].tail(1)['rsi'].values[0]
        self.log.debug(f'After Calculating RSI. all_data.shape SOBE: {self._all_data["SOBE"].shape}')

      self.log.debug(f'regression_times SOBE: {self._regression_times["SOBE"]}')
      if (self._regression_times['SOBE'] is not None) and (self._regression_times['SOBE'] > 0):
        self._all_data['SOBE'], _ = utils.regresstion_times(self._all_data['SOBE'], self._numeric_features['SOBE'], self._regression_times['SOBE'], last_one=True)

    return rsi

  def calc_sl_pnl(self, strategy, actual_value, margin):
    take_profit_value = 0.0
    stop_loss_value = 0.0
    if strategy.startswith('CAI'):  # Short
      take_profit_value = actual_value * (1 - margin / 100)
      stop_loss_value = actual_value * (1 + (margin * myenv.stop_loss_multiplier) / 100)
    elif strategy.startswith('SOBE'):  # Long
      take_profit_value = actual_value * (1 + margin / 100)
      stop_loss_value = actual_value * (1 - (margin * myenv.stop_loss_multiplier) / 100)
    return take_profit_value, stop_loss_value

  def validate_short_or_long(self, strategy):
    return strategy.startswith('SOBE') or strategy.startswith('CAI')

  def is_long(self, strategy):
    return strategy.startswith('SOBE')

  def is_short(self, strategy):
    return strategy.startswith('CAI')

  def log_info(self, purchased, open_time, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance, take_profit,
               stop_loss, target_margin, strategy):
    if isinstance(open_time, np.datetime64):
      _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    else:
      _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')
    if purchased:
      msg = f'*PURCHASED*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Strategy: {strategy} - Target Margin: {target_margin:.2f}% '
      msg += f'- Purchased Price: $ {purchase_price:.6f} - Actual Price: $ {actual_price:.6f} - Margin Operation: {100*margin_operation:.2f}% - Amount invested: $ {amount_invested:.2f} '
      msg += f'- PnL: $ {profit_and_loss:.2f} - Take Profit: $ {take_profit:.6f} - Stop Loss: $ {stop_loss:.6f} - Balance: $ {balance:.2f}'
    else:
      msg = f'*NOT PURCHASED*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Actual Price: $ {actual_price:.6f} - Balance: $ {balance:.2f}'
    self.log.info(f'{msg}')
    sm.send_status_to_telegram(msg)

  def log_buy(self, open_time, strategy, purchase_price, amount_invested, balance, target_margin, take_profit, stop_loss, rsi, restored=False):
    if isinstance(open_time, np.datetime64):
      _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    else:
      _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')

    status = ''
    if restored:
      status = '-RESTORED'

    msg = f'*BUYING{status}*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Strategy: {strategy} - Target Margin: {target_margin:.2f}% - '
    msg += f'Purchased Price: $ {purchase_price:.6f} - Amount invested: $ {amount_invested:.2f} - Take Profit: $ {take_profit:.6f} - '
    msg += f'Stop Loss: $ {stop_loss:.6f} - RSI: {rsi:.2f} - Balance: $ {balance:.2f}'
    self.log.info(f'{msg}')
    sm.send_to_telegram(msg)

  def log_selling(self, open_time, strategy, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance, take_profit,
                  stop_loss, target_margin, rsi):
    if isinstance(open_time, np.datetime64):
      _open_time = pd.to_datetime(open_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    else:
      _open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')

    sum_pnl = utils.get_sum_pnl()
    msg = f'*SELLING*: Symbol: {self._symbol}_{self._interval} - Open Time: {_open_time} - Strategy: {strategy} - Target Margin: {target_margin:.2f}% '
    msg += f'- Purchased Price: $ {purchase_price:.6f} - Actual Price: $ {actual_price:.6f} - Margin Operation: {100*margin_operation:.2f}% - Amount invested: $ {amount_invested:.2f} '
    msg += f'- Take Profit: $ {take_profit:.6f} - Stop Loss: $ {stop_loss:.6f} - RSI: {rsi:.2f} - Balance: $ {balance:.2f} - PnL Operation: $ {profit_and_loss:.2f} - Sum PnL: $ {sum_pnl:.2f}'
    self.log.info(f'{msg}')
    sm.send_to_telegram(msg)

  def predict_strategy(self, strategy: str):
    result_strategy, target_margin = 'ESTAVEL', 0.0
    self.log.info(f'Start Predicting Strategy for: {strategy}')
    if strategy in self._experiment:
      if self._experiment[strategy] is None:
        return result_strategy, target_margin

      df_to_predict = self._all_data[strategy].tail(1)
      self.log.debug(f'Data input to Predict for: {strategy}:\n{df_to_predict.to_dict(orient="records")}')

      df_predict = self._experiment[strategy].predict_model(self._model[strategy], df_to_predict, verbose=self._verbose)

      prediction_label = df_predict['prediction_label'].values[0]
      self.log.info(f'Strategy Predicted for: {strategy}: {prediction_label}')

      if self.validate_short_or_long(prediction_label):
        result_strategy = prediction_label.split('_')[0]
        target_margin = float(prediction_label.split('_')[1])

    return result_strategy, target_margin

  def run(self):
    #self.log.info(f'Columns CAI: {self._kline_features["CAI"]}')

    self.log.info(f'Start _data_preprocessing...')
    self._data_preprocessing()
    self.log.info(f'Start _feature_engineering...')
    self._feature_engineering()
    self.log.info(f'Start _load_model...')
    self._load_model()

    cont = 0
    cont_aviso = 101

    purchased = False
    purchase_price = 0.0
    actual_price = 0.0
    amount_invested = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    profit_and_loss = 0.0
    strategy = ''
    margin_operation = 0.0
    target_margin = 0.0
    rsi = {'CAI': 0.0, 'SOBE': 0.0}
    latest_closed_candle_open_time_aux = None

    balance = utils.get_account_balance()

    ledger = utils.get_latest_operation(self._symbol, self._interval)
    if (ledger is not None) and (ledger.operation == 'BUY'):
      purchased = True
      purchase_price = ledger.purchase_price
      take_profit = ledger.take_profit
      stop_loss = ledger.stop_loss
      amount_invested = ledger.amount_invested
      rsi = {'CAI': ledger.rsi, 'SOBE': ledger.rsi}
      target_margin = ledger.target_margin
      strategy = ledger.strategy
      latest_closed_candle_open_time_aux = pd.to_datetime(ledger.operation_date)
      self.log_buy(latest_closed_candle_open_time_aux, strategy, purchase_price, amount_invested, balance, target_margin, take_profit, stop_loss, rsi[strategy], True)

    error = False
    first_loop = True
    while True:
      try:
        error = False
        # Update data
        balance = utils.get_account_balance()
        actual_price, latest_closed_candle_open_time = self.update_data_from_web()

        # Apply predict only on time per interval
        if (not purchased) and (latest_closed_candle_open_time_aux != latest_closed_candle_open_time) and (balance > 0):
          latest_closed_candle_open_time_aux = latest_closed_candle_open_time
          if first_loop:
            first_loop = False
            continue

          rsi = self.feature_engineering_on_loop()
          strategy, target_margin = self.predict_strategy('CAI')
          self.log.info(f'Predicted Strategy: {strategy} - Target Margin: {target_margin:.2f}%')
          if strategy != 'CAI':
            strategy, target_margin = self.predict_strategy('SOBE')
            if strategy != 'SOBE':
              strategy = 'ESTAVEL'

          if self.validate_short_or_long(strategy) and strategy in self._all_data:  # If true, BUY
            purchased = True
            purchase_price = actual_price
            take_profit, stop_loss = self.calc_sl_pnl(strategy, purchase_price, target_margin)

            # Lock Thread to register BUY
            self._mutex.acquire()
            amount_invested, balance = utils.get_amount_to_invest(register=True)
            ledger_params = utils.get_params_operation(latest_closed_candle_open_time, self._symbol, self._interval, 'BUY', target_margin, amount_invested, take_profit, stop_loss,
                                                       purchase_price, rsi[strategy], 0.0, 0.0, 0.0, strategy, balance)
            utils.register_operation(ledger_params)
            self._mutex.release()
            # End Lock

            self.log_buy(latest_closed_candle_open_time_aux, strategy, purchase_price, amount_invested, balance, target_margin, take_profit, stop_loss, rsi[strategy])
            self.log.debug(f'\nPerform BUY: Strategy: {strategy}\nActual Price: $ {actual_price:.6f}\nPurchased Price: $ {purchase_price:.6f}'
                           f'\nAmount Invested: $ {amount_invested:.2f}\nTake Profit: $ {take_profit:.6f}\nStop Loss: $ {stop_loss:.6f}\n'
                           f'\nPnL: $ {profit_and_loss:.2f}\nTarget Margin: {target_margin:.2f}%\nRSI: {rsi[strategy]:.2f}\nBalance: $ {balance:.2f}')
            continue

        if purchased:  # and (operation.startswith('SOBE') or operation.startswith('CAI')):
          perform_sell = False
          if self.is_long(strategy):
            margin_operation = (actual_price - purchase_price) / purchase_price
            if ((actual_price >= take_profit) or (actual_price <= stop_loss)):  # Long ==> Sell - Take Profit / Stop Loss
              perform_sell = True
          elif self.is_short(strategy):
            margin_operation = (purchase_price - actual_price) / purchase_price
            if ((actual_price <= take_profit) or (actual_price >= stop_loss)):  # Short ==> Sell - Take Profit / Stop Loss
              perform_sell = True

          profit_and_loss = amount_invested * margin_operation

          if perform_sell:  # Register Sell
            self.log.debug(f'\nPerform SELL: Strategy: {strategy}\nActual Price: $ {actual_price:.6f}\nPurchased Price: $ {purchase_price:.6f}'
                           f'\nAmount Invested: $ {amount_invested:.2f}\nTake Profit: $ {take_profit:.6f}\nStop Loss: $ {stop_loss:.6f}'
                           f'\nMargin Operation: {100*margin_operation:.2f}%\nPnL: $ {profit_and_loss:.2f}\nTarget Margin: {target_margin:.2f}%'
                           f'\nRSI: {rsi[strategy]:.2f}\nBalance: $ {balance:.2f}')

            # Lock Thread to register SELL
            self._mutex.acquire()
            balance = utils.register_account_balance(amount_invested + profit_and_loss)
            ledger_params = utils.get_params_operation(latest_closed_candle_open_time, self._symbol, self._interval, 'SELL', target_margin, amount_invested, take_profit, stop_loss,
                                                       purchase_price, rsi[strategy], actual_price, profit_and_loss, margin_operation, strategy, balance)
            utils.register_operation(ledger_params)
            self._mutex.release()
            # End Lock

            self.log_selling(latest_closed_candle_open_time, strategy, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance,
                             take_profit, stop_loss, target_margin, rsi[strategy])
            # Reset variables
            purchased, purchase_price, amount_invested, take_profit, stop_loss, profit_and_loss, margin_operation, target_margin = (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      except Exception as e:
        if self._mutex.locked():
          self._mutex.release()
        self.log.exception(e)
        sm.send_status_to_telegram('ERROR: ' + str(e))
        error = True
      finally:
        time.sleep(myenv.sleep_refresh)
        cont += 1
        cont_aviso += 1
        if cont_aviso > 100 and not error:
          cont_aviso = 0
          self.log_info(purchased, latest_closed_candle_open_time_aux, purchase_price, actual_price, margin_operation, amount_invested, profit_and_loss, balance,
                        take_profit, stop_loss, target_margin, strategy)
