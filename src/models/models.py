import sqlalchemy
print(f'sqlalchemy: {sqlalchemy.__version__}')
from sqlalchemy import create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from typing import Optional
from datetime import datetime
import sys

import src.myenv as myenv

url = f'sqlite:///{sys.path[0]}/db/mg_trader.db'
engine = create_engine(url, echo=False)

# https://docs.sqlalchemy.org/en/20/orm/declarative_styles.html


class Base(DeclarativeBase):
  pass


class AccountBalance(Base):
  __tablename__ = 'account_balance'

  id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
  balance: Mapped[float]
  created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Ledger(Base):
  __tablename__ = 'ledger'
  # operation_date;symbol;interval;operation;amount_invested;balance;take_profit;stop_loss;purchase_price;sell_price;PnL;rsi;status

  id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
  operation_date: Mapped[datetime]
  symbol: Mapped[str]
  interval: Mapped[str]
  operation: Mapped[str]
  target_margin: Mapped[float]
  amount_invested: Mapped[float]
  take_profit: Mapped[float]
  stop_loss: Mapped[float]
  purchase_price: Mapped[float]
  sell_price: Mapped[Optional[float]]
  pnl: Mapped[Optional[float]]
  rsi: Mapped[float]
  margin_operation: Mapped[float]
  balance: Mapped[float]
  created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

  # __mapper_args__ = {"sqlite_autoincrement": True, }
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
with Session() as session:
  if len(session.query(AccountBalance.id).all()) == 0:
    ab = AccountBalance(balance=myenv.initial_amount_balance)
    print(f'initializing account_balance. Value: {ab.balance}')
    session.add(ab)
    session.commit()
