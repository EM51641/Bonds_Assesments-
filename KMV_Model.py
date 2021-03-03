import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import timedelta, date
import time
from yahoo_fin.options import *
import yfinance as yf
from pandas import ExcelWriter
from yahoo_fin import stock_info as si
from arch import arch_model
import math
import random
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters
import mplfinance as mpf
from numba import jit
from scipy.optimize import minimize
from scipy.optimize import leastsq
register_matplotlib_converters()
yf.pdr_override()


class OptionTools:

    def __init__(self):
        pass
    
    def Load_data(self,Ticker):
        B_Sheet = si.get_balance_sheet(Ticker,yearly = False )
        Total_assets = B_Sheet.loc['totalAssets'][-1]
        try:
            Total_Debt = B_Sheet.loc['longTermDebt'].dropna()[-1]/2 + B_Sheet.loc['shortLongTermDebt'].dropna()[-1]
        except:
            Total_Debt = B_Sheet.loc['longTermDebt'].dropna()[-1]/2
            print('NA short_term')
            
        A = si.get_quote_table(Ticker)['Market Cap']
        
        if A[-1] == 'B':
            MKP = float(A[:-1]) * 10**9
        elif A[-1] == 'M' :
            MKP = float(A[:-1]) * 10**6
        else:
            MKP = float(A[:-1]) * 10**12
            
        return Total_assets,Total_Debt,MKP
    
    def Load_sp(self,Ticker,Total_Debt,Total_assets):
        d= pd.DataFrame()
        d=wb.DataReader(Ticker,data_source='yahoo',start='2010-12-01')['Adj Close']
        stdv = d.pct_change().dropna().std() * 252**(1/2) 
        return stdv
    
    def Calibration(self,params,args):
        sigma,Assets = params
        Market_cap,Debts,sigma2 = args
        ALS = EuropeanCall(Assets,sigma,Debts,5,0.0232,0)
        delta_ = ALS.delta
        Price = ALS.price
        Diff1 = (Market_cap - Price)
        Diff2 = (sigma2 * Market_cap - delta_ * sigma * Assets )
        return Diff1,Diff2
    
    def Calibration20(self,params,args):
        sigma,Assets = params
        Market_cap,Debts,sigma2 = args
        d1 = (np.log(Assets/Debts) + (0.0232+.5*sigma**2))/sigma
        d2 = d1 - sigma
        Diff1 = Market_cap - (Assets*norm.cdf(d1)-Debts*norm.cdf(d2)*np.exp(-0.0232))
        Diff2 = sigma2*Market_cap - norm.cdf(d1) * sigma * Assets
        f = np.zeros(2)
        f[0] = Diff1**2
        f[1] = Diff2**2
        return -Diff2
       
class EuropeanCall:

    def call_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration,risk_free_rate,q
            ):
        asset_volatility = np.array(asset_volatility)
        asset_price = np.array(asset_price)
        strike_price = np.array(strike_price)
        risk_free_rate = np.array(risk_free_rate)
        b = np.exp(-risk_free_rate*time_to_expiration)
        x1 = np.log(asset_price/(strike_price)) + .5*(asset_volatility*asset_volatility+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1

    def call_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        asset_volatility = np.array(asset_volatility)
        asset_price = np.array(asset_price)
        strike_price = np.array(strike_price)
        risk_free_rate = np.array(risk_free_rate)
        b  = np.exp(-risk_free_rate*time_to_expiration)
        x1 = np.log(asset_price/(strike_price))+(.5*(asset_volatility**2)+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z1 = z1*asset_price*math.exp(-q*time_to_expiration)
        x2 = np.log(asset_price/(strike_price)) - (.5*(asset_volatility**2)-risk_free_rate+q)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(x2)
        z2 = b*strike_price*z2
        return z1 - z2

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.q = q
        self.price = self.call_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)
        self.delta = self.call_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)  
 
if __name__ == '__main__':
    Ticker = 'TSLA'
    Assets,Debts,Market_cap = OptionTools().Load_data(Ticker)
    Assets = Market_cap
    sigma2 = OptionTools().Load_sp(Ticker,Debts,Assets)
    Equity = Market_cap
    args2 = [Market_cap,Debts,sigma2]
    res = leastsq(OptionTools().Calibration,[sigma2,Assets],args2)
    volatilitly,Assets = res[0][0],res[0][1]
    DD = 1-EuropeanCall(Assets,volatilitly,Debts,5,0.0232,0).delta
    #--------------------------
    print('Risk of Default :',DD)
    print('Asset Volatility :',volatilitly)
    print('Equity Volatility :',sigma2)
    print('Asset Value :',Assets)
    print('Market cap :',Market_cap)
        
