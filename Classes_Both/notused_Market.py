from dataclasses import dataclass

@dataclass
class Market:
    sigma: float
    r: float
    dividends: list
    price: float