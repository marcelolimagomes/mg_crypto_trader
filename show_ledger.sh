#!/bin/bash

while true; do
  # Your command goes here
  #  echo 'OPERATION | SYMBOL | INTERVAL | PnL | BALANCE | DATE OPERATION'
  # sqlite3 db/mg_trader.db 'SELECT operation, symbol, interval, round(pnl, 2), round(balance,2), strftime("%Y-%m-%d %H:%M:%S", "created_at") FROM ledger'
  # echo ''

  echo 'SYMBOL | INTERVAL | TOTAL PnL | NUMBER SELLs'
  sqlite3 db/mg_trader.db 'SELECT symbol, interval, sum(round(pnl, 2)), count(1) FROM ledger WHERE operation = "SELL" and strategy = "LONG" GROUP BY symbol, interval ORDER BY symbol, sum(pnl)'
  echo ''

  echo 'SYMBOL | TOTAL PnL | NUMBER SELLs'
  sqlite3 db/mg_trader.db 'SELECT symbol, sum(round(pnl, 2)), count(1) FROM ledger WHERE operation = "SELL" and strategy = "LONG" GROUP BY symbol ORDER BY sum(pnl)'
  echo ''

  echo 'OPERATION | TOTAL PnL | NUMBER OPERATIONS'
  sqlite3 db/mg_trader.db 'SELECT operation, sum(round(pnl, 2)), count(1) FROM ledger WHERE strategy = "LONG" GROUP BY operation ORDER BY sum(pnl)'
  echo ''

  echo 'OPERATION | TOTAL PnL | NUMBER OPERATIONS'  
  sqlite3 db/mg_trader.db 'SELECT operation, strategy, sum(round(pnl, 2)) as soma, count(1) FROM ledger WHERE operation="SELL" GROUP BY operation, strategy ORDER BY sum(pnl)'
  echo ''

  echo 'TOTAL OPERATIONS IN BUY'
  sqlite3 db/mg_trader.db 'SELECT (SELECT count(1) FROM ledger WHERE operation = "BUY" and strategy = "LONG" GROUP BY operation) - (SELECT count(1) FROM ledger WHERE operation = "SELL" and strategy = "SELL" GROUP BY operation)'

  echo $(date)

  # Sleep for 5 seconds
  sleep 20
done




