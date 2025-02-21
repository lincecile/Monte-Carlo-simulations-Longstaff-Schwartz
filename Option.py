import datetime as dt
from dataclasses import dataclass
@dataclass
class Option:
    pricing_date : dt.datetime
    maturity_date : dt.datetime
    maturity : dt.datetime = (maturity_date - pricing_date).days
    strike : float
    call : bool = True
    american : bool = False