import sys
from src.batch_robo_trader import BatchRoboTrader
from src.utils import *


def main(args):
  # Boolean arguments
  verbose = False
  update_data_from_web = False

  # Single arguments
  start_date = '2023-01-01'
  log_level = logging.INFO
  prediction_mode = "ml"

  for arg in args:
    # Boolean arguments
    if (arg.startswith('-verbose')):
      verbose = True

    if (arg.startswith('-update-data-from-web')):
      update_data_from_web = True

    # Single arguments
    if (arg.startswith('-start-date=')):
      start_date = arg.split('=')[1]

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

  brt = BatchRoboTrader(
      verbose,
      start_date,
      prediction_mode,
      update_data_from_web,
      log_level)
  brt.run()


if __name__ == '__main__':
  main(sys.argv[1:])
