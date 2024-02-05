import sys

interval_list = ['1m', '5m', '15m', '30m', '1h']

float_kline_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

integer_kline_cols = ['number_of_trades', 'ignore']

date_kline_cols = ['open_time', 'close_time']

# must be that order
all_klines_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                   'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

all_cols = all_klines_cols + ['symbol']
all_cols.remove('ignore')

all_index_cols = ['open_time', 'close']

essential_cols = ['close']

data_numeric_fields = ['open', 'high', 'low', 'volume', 'close']
date_features = ['open_time']
use_cols = date_features + data_numeric_fields
numeric_features = data_numeric_fields + ['rsi']
datadir = sys.path[0] + '/data'
logdir = sys.path[0] + '/logs'
train_log_filename = 'batch_training.log'
batch_robo_log_filename = 'batch_robo.log'
main_log_filename = 'main_process_trader.log'
robo_log_filename = 'robo.log'
modeldir = sys.path[0] + '/src/models'
label = 'status'
stop_loss = 1.0
regression_times = 24 * 30 * 2  # horas
times_regression_profit_and_loss = 24
n_jobs = -1
train_size = 0.7
estimator = 'xgboost'
symbol = 'BTCUSDT'
saldo_inicial = 100
stop_loss_multiplier = 5
stop_loss_range_multiplier = 7
range_min_rsi_end = 60
range_max_rsi_start = 68


# :::>>>
default_amount_to_invest = 150.00

min_amount_to_invest = 6.00

min_pnl_to_include_on_best_params = 100.0

initial_amount_balance = 10000
rows_to_train = 50000

rows_to_index_train = 1000

days_to_validate_train = 15  # Tree months

sleep_refresh = 5  # seconds

producao = True

telegram_key = []

min_rsi = 30
max_rsi = 70
p_ema = 200

retrain_last_results = 100

asset_balance_currency = 'USDT'
max_purchase_attemps = 60
recv_window = 10000
