from src.train_best_model import TrainBestModel
from src.batch_train_ml import BatchTrainML
from src.batch_train_index import BatchTrainIndex
from src.utils import *

import src.myenv as myenv

import datetime
import logging
import sys

logger = None


def configure_log(log_level):
  log_file_path = os.path.join(myenv.logdir, myenv.train_log_filename)
  logger = logging.getLogger('training_logger')
  logger.propagate = False
  logger.setLevel(log_level)

  fh = logging.FileHandler(log_file_path, mode='a', delay=True)
  fh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
  fh.setLevel(log_level)

  sh = logging.StreamHandler()
  sh.setFormatter(logging.Formatter(f'[%(asctime)s] - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
  sh.setLevel(log_level)

  logger.addHandler(fh)
  logger.addHandler(sh)
  return logger


def main(args):
  # Boolean arguments
  update_data_from_web = False
  calc_rsi = False
  use_gpu = False
  normalize = False
  verbose = False
  use_all_data_to_train = False
  revert = False
  no_tune = False
  update_database = False
  save_model = False
  feature_selection = False
  combine_features = False

  # Single arguments
  start_train_date = '2010-01-01'
  start_test_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
  fold = 3
  n_jobs = myenv.n_jobs
  n_threads = 1
  log_level = logging.DEBUG

  # List arguments
  symbol_list = get_symbol_list()
  interval_list = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
  estimator_list = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr',
                    'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost']
  target_margin_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  numeric_features_list = prepare_numeric_features_list(myenv.data_numeric_fields)
  times_regression_PnL_list = [6, 12, 24]
  regression_times_list = [24, 360, 720]
  regression_features_list = combine_list(myenv.data_numeric_fields)
  prediction_mode = 'ml'

  range_min_rsi = 30
  range_max_rsi = 70
  range_p_ema = [50, 250]

  for arg in args:
    # Boolean arguments
    if (arg.startswith('-update-data-from-web')):
      update_data_from_web = True
    if (arg.startswith('-calc-rsi')):
      calc_rsi = True
    if (arg.startswith('-use-gpu')):
      use_gpu = True
    if (arg.startswith('-normalize')):
      normalize = True
    if (arg.startswith('-verbose')):
      verbose = True
    if (arg.startswith('-use-all-data-to-train')):
      use_all_data_to_train = True
    if (arg.startswith('-revert')):
      revert = True
    if (arg.startswith('-no-tune')):
      no_tune = True
    if (arg.startswith('-update-database')):
      update_database = True
    if (arg.startswith('-save-model')):
      save_model = True
    if (arg.startswith('-feature-selection')):
      feature_selection = True
    if (arg.startswith('-combine-features')):
      combine_features = True

    # Single arguments
    if (arg.startswith('-start-train-date=')):
      start_train_date = arg.split('=')[1]

    if (arg.startswith('-start-test-date=')):
      start_test_date = arg.split('=')[1]

    if (arg.startswith('-fold=')):
      fold = int(arg.split('=')[1])

    if (arg.startswith('-n-jobs=')):
      n_jobs = int(arg.split('=')[1])

    if (arg.startswith('-n-trheads=')):
      n_threads = int(arg.split('=')[1])

    if (arg.startswith('-log-level=DEBUG')):
      log_level = logging.DEBUG

    if (arg.startswith('-log-level=WARNING')):
      log_level = logging.WARNING

    if (arg.startswith('-log-level=INFO')):
      log_level = logging.INFO

    if (arg.startswith('-log-level=ERROR')):
      log_level = logging.ERROR

    if (arg.startswith('-prediction-mode=')):
      prediction_mode = arg.split('=')[1]

    if (arg.startswith('-range-min-rsi=')):
      range_min_rsi = int(arg.split('=')[1])

    if (arg.startswith('-range-max-rsi=')):
      range_max_rsi = int(arg.split('=')[1])

    # List arguments
    if (arg.startswith('-symbol-list=')):
      aux = arg.split('=')[1]
      symbol_list = aux.split(',')

    if (arg.startswith('-interval-list=')):
      aux = arg.split('=')[1]
      interval_list = aux.split(',')

    if (arg.startswith('-estimator-list=')):
      aux = arg.split('=')[1]
      estimator_list = aux.split(',')

    if (arg.startswith('-target-margin-list=')):
      aux = arg.split('=')[1]
      target_margin_list = aux.split(',')

    if (arg.startswith('-numeric-features=')):
      aux = arg.split('=')[1]
      aux += 'ema_XXXp,ema_200p' if aux == '' else ',ema_XXXp,ema_200p'
      if ('-calc-rsi' in args) and ('rsi' not in aux):
        aux += ',rsi'
      if combine_features:
        numeric_features_list = prepare_numeric_features_list(aux.split(','))
      else:
        numeric_features_list = [aux]

    if (arg.startswith('-regression-PnL-list=')):
      aux = arg.split('=')[1]
      times_regression_PnL_list = aux.split(',')

    if (arg.startswith('-regression-times-list=')):
      aux = arg.split('=')[1]
      regression_times_list = aux.split(',')

    if (arg.startswith('-regression-features=')):
      aux = arg.split('=')[1]
      if combine_features:
        regression_features_list = combine_list(aux.split(','))
      else:
        regression_features_list = [aux]

    if (arg.startswith('-range-p-ema=')):
      aux = arg.split('=')[1]
      range_p_ema = aux.split(',')

  logger = configure_log(log_level)

  if '-train-best-model' in args:
    print('Starting Train Best Model...')
    best = TrainBestModel(verbose, log_level)
    best.run()
    sys.exit(0)

  if update_database:
    logger.info('start_batch_training: Updating database...')
    download_data(save_database=True, parse_dates=False)
    logger.info('start_batch_training: Database updated...')

  if prediction_mode == 'ml':
    bt = BatchTrainML(
        update_data_from_web,
        calc_rsi,
        use_gpu,
        normalize,
        verbose,
        use_all_data_to_train,
        revert,
        no_tune,
        feature_selection,
        combine_features,
        save_model,
        start_train_date,
        start_test_date,
        fold,
        n_jobs,
        n_threads,
        log_level,
        symbol_list,
        interval_list,
        estimator_list,
        target_margin_list,
        numeric_features_list,
        times_regression_PnL_list,
        regression_times_list,
        regression_features_list)
    bt.run()
  elif prediction_mode == 'index':
    bt = BatchTrainIndex(
        update_data_from_web,
        calc_rsi,
        verbose,
        log_level,
        symbol_list,
        interval_list,
        target_margin_list,
        range_min_rsi,
        range_max_rsi,
        range_p_ema)
    bt.run()
  else:
    raise ValueError(f'Invalid prediction mode: {prediction_mode}')


if __name__ == '__main__':
  main(sys.argv[1:])
