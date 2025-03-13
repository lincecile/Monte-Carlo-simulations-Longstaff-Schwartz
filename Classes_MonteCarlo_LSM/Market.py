from dataclasses import dataclass
from Classes_MonteCarlo_LSM.module_brownian import Brownian
import numpy as np
import pandas as pd
@dataclass
class Market:
    sigma: float
    r: float
    dividends: list
    price: float

    
               