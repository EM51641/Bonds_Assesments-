import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from arch import arch_model
from scipy.optimize import minimize 
from random import random
import statsmodels.api as sm
from statsmodels.formula.api import ols
import quandl
from numba import jit
from datetime import timedelta, date
import time
from yahoo_fin.options import *
import yfinance as yf
from pandas import ExcelWriter
from yahoo_fin import stock_info as si
import math
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters
import mplfinance as mpf
from scipy.optimize import leastsq
register_matplotlib_converters()
import QuantLib as ql
yf.pdr_override()

class Initialize_parameters:
    
    def __init__(self):
        pass
    
    def kappa_sigma_theta_initial_estimators(self,dt,cond_v):
        DF=pd.DataFrame(cond_v).dropna()
        dif=np.array(DF.iloc[1:].values-DF.iloc[:-1].values)
        rs=np.array(DF.iloc[:-1].values)
        Y=(dif/np.sqrt(rs))
        Y=pd.DataFrame(Y)
        Y.columns=['Y']
        B1=dt/np.sqrt(rs)
        B1=pd.DataFrame(B1)
        B1.columns=['Beta1']
        B2=dt*np.sqrt(rs)
        B2=pd.DataFrame(B2)
        B2.columns=['Beta2']
        X=(B1.join(B2))
        modl=sm.OLS(Y,X)
        resl=modl.fit()
        kappa=-resl.params[-1]
        theta=resl.params[0]/kappa
        xi=np.std(resl.resid)/np.sqrt(dt)
        return  kappa,theta,xi
    
    @staticmethod
    @jit(nopython=True)
    def Monte_Carlo(cond_v, kappa, theta, xi,dt,n):
        r  = np.zeros(n)
        r[0] = cond_v
        for t in range(1,n):
            r[t] = r[t-1]+kappa*(theta - r[t-1])*dt +xi * np.sqrt(r[t-1])*np.sqrt(dt)*np.random.normal(0, 1)
        return r
    
    def LogL(self,params,args):
        kappa,theta,xi = params
        dt ,n,rfree = args
        c = 2*kappa/((xi**2)*(1-np.exp(-kappa*dt)))
        q = 2*kappa*(theta/xi**2)-1
        u = c*np.exp(-kappa*dt)*rfree[:-1].values
        v = c*rfree[1:].values
        z = 2*np.sqrt(u*v)
        bf = scipy.special.ive(q,z)
        lnL= -(n-1)* np.log(c) + np.sum(u + v - 0.5*q*np.log(v/u) - np.log(bf) - z)
        return lnL
    
    
    def MCR(self,cond_v, kappa, theta, xi,dt,n,J):
        rm = pd.DataFrame()
        for t in range(0,J):
            rm[t] = self.Monte_Carlo(cond_v, kappa, theta, xi,dt,n)
        return rm
        
if __name__ == '__main__':
    dt    = 1/2
    rfree = quandl.get("ML/BBBEY", authtoken="bBxaD71sAGrij1mxHsys")
    rfree = rfree.loc['1995-01-01':].resample('6M').last()/100
    rfree = pd.DataFrame(rfree)
    Kappa,theta,xi = Initialize_parameters().kappa_sigma_theta_initial_estimators(dt,rfree)
    args = [dt,len(rfree),rfree]
    res = minimize(Initialize_parameters().LogL,[Kappa,theta,xi],args,method='SLSQP')
    p=res.x
    
class Fisher_Black_Call:

    def call_delta(
        self, asset_price, strike_price,
        time_to_expiration,risk_free_rate,Duration,Kappa,Theta,xi,dt
            ):
        asset_volatility = self.asset_volatility(Duration,Kappa,Theta,xi,time_to_expiration, risk_free_rate,dt)
        asset_price = np.array(asset_price)
        strike_price = np.array(strike_price)
        risk_free_rate = np.array(risk_free_rate)
        b = np.exp(-risk_free_rate*time_to_expiration)
        x1 = np.log(asset_price/(strike_price)) + .5*(asset_volatility*asset_volatility)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1

    def call_price(
        self, asset_price, strike_price,
        time_to_expiration, risk_free_rate,Duration,Kappa,Theta,xi,dt
            ):
        asset_volatility = self.asset_volatility(Duration,Kappa,Theta,xi,time_to_expiration, risk_free_rate,dt)
        asset_price = np.array(asset_price)
        strike_price = np.array(strike_price)
        risk_free_rate = np.array(risk_free_rate)
        b  = np.exp(-risk_free_rate*time_to_expiration)
        x1 = np.log(asset_price/(strike_price))+(.5*(asset_volatility**2))*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z1 = z1*asset_price
        x2 = np.log(asset_price/(strike_price)) - (.5*(asset_volatility**2))*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(x2)
        z2 = b*strike_price*z2
        return z1 - z2
    
    def asset_volatility(self,Duration,Kappa,Theta,xi,time_to_expiration,risk_free_rate,dt):
        Expected_Variance = risk_free_rate*(xi**2/Kappa)*(np.exp(-Kappa*time_to_expiration*dt)-np.exp(-2*Kappa*time_to_expiration*dt)) + \
                            (theta * (xi**2)/2*Kappa)*(1-np.exp(-Kappa*time_to_expiration*dt))**2
        Implied_volatility = Duration * np.sqrt(Expected_Variance)
        return Implied_volatility
        
    def __init__(
        self, asset_price, strike_price,
        time_to_expiration, risk_free_rate,Duration,Kappa,Theta,xi,dt):
        self.asset_price = asset_price
        self.volatility = self.asset_volatility(Duration,Kappa,Theta,xi,time_to_expiration,risk_free_rate,dt)
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.price = self.call_price(asset_price, strike_price, time_to_expiration, risk_free_rate,Duration,Kappa,Theta,xi,dt)
        self.delta = self.call_delta(asset_price, strike_price, time_to_expiration, risk_free_rate,Duration,Kappa,Theta,xi,dt)
        
        
class Bond_Evaluation:
    
    def Bond_Pricer(self,coupon,frequency,dayCount,price,T0,T1):
        bond = self.bond_function(coupon,frequency,dayCount,T0,T1)
        Yield = bond.bondYield(price, dayCount, ql.Compounded, ql.Annual)
        Price = bond.dirtyPrice(Yield, dayCount, ql.Compounded, ql.Annual)
        return Price,Yield
        
    def Duration_computation(self,coupon,frequency,dayCount,price,T0,T1):
        bond = self.bond_function(coupon,frequency,dayCount,T0,T1)
        yieldm = bond.bondYield(price, dayCount, ql.Compounded, ql.Annual)
        rate = ql.InterestRate(yieldm, ql.ActualActual(), ql.Compounded, ql.Annual)
        cvx = ql.BondFunctions.convexity(bond, rate)
        Duration = ql.BondFunctions.duration(bond, rate)
        DS =  -Duration * 0.01/1.01 + .5 * cvx * (.01)**2
        return Duration,DS
    
    def bond_function(self,coupon,frequency,dayCount,T0,T1):
        start,maturity = self.Maturity_start_calculator(T0,T1)
        bond = ql.FixedRateBond(0, ql.TARGET(), 100.0, start, maturity, ql.Period(frequency), [coupon], dayCount)
        return bond
    
    def Superflous(self,freq,T0,T1):
        Total = self.delta_computator(T0,T1)[1]
        Superfl = np.maximum(abs(Total - int(Total) - 1/freq),0)
        return Superfl
    
    def bond_price_Given_Yield_Curve(self,par, coupon, freq,T0,T1,Kappa, theta, xi,dt,Benchmark_yield):
        T = self.delta_computator(T0,T1)[1]
        J = 1000
        rate = self.Zero_volatility_Yield_Structure_caller(Benchmark_yield,Kappa, theta, xi,dt,freq,T0,T1,J)
        #rate = rate.values.reshape(len(rate.columns),len(rate))
        Superflous = self.Superflous(freq,T0,T1)
        freq = float(freq)
        periods = T*freq
        coupon = coupon*par/freq
        r = rate #[rate[i] for i in range(int(periods))]
        dt = [((i+1)/freq) for i in range(int(periods))]
        Lst_price = []
        for j in range(len(rate.columns)):
            price = sum([coupon/(1+(r.iloc[int(t*freq - 1),j])/freq)**(freq*(t-Superflous)) for t in dt]) + \
                par/(1+(r.iloc[-1,j])/freq)**(freq*(T-Superflous))
            Lst_price.append(price)
        Lst_price = np.array(Lst_price).reshape(1,len(Lst_price))[0]
        price = np.mean(Lst_price)
        return price
    
    def Maturity_start_calculator(self,T0,T1):
        delta = self.delta_computator(T0,T1)[0]
        start = ql.Date().todaysDate()
        maturity = start + ql.Period(delta.days, ql.Days)
        return start,maturity
    
    def delta_computator(self,T0,T1):
        delta = T1 - T0
        T = round(delta.days/360,3)
        return delta,T
    
    def Z_Spread_finder(self,z,args):
        Real_Price,par, T, rate, coup , freq,T0,T1 = args
        Superflous = self.Superflous(freq,T0,T1)
        freq = float(freq)
        periods = T*freq
        coupon = coup*par/freq
        r = rate #[rate[i] for i in range(int(periods))]
        dt = [((i+1)/freq) for i in range(int(periods))]
        Lst_price = []
        for j in range(len(rate.columns)):
            price = sum([coupon/(1+(r.iloc[int(t*freq - 1),j]+z)/freq)**(freq*(t-Superflous)) for t in dt]) + \
                par/(1+(r.iloc[-1,j]+z)/freq)**(freq*(T-Superflous))
            Lst_price.append(price)
        Lst_price = np.array(Lst_price).reshape(1,len(Lst_price))[0]
        return np.sum(((Lst_price-Real_Price)/Real_Price)**2)
    
    def Z_spread_Optimizor(self,coupon,frequency,dayCount,price,Benchmark_yield,T0,T1,par, freq,Kappa, theta, xi,dt):
        T = self.delta_computator(T0,T1)[1]
        J = 1000
        Real_Price = self.Bond_Pricer(coupon,frequency,dayCount,price,T0,T1)[0]
        #J = 1
        #xi = 0
        rate = self.Zero_volatility_Yield_Structure_caller(Benchmark_yield, Kappa, theta,xi,dt,freq,T0,T1,J)
        #rate = rate.values.reshape(len(rate.columns),len(rate))
        args = [Real_Price,par, T, rate, coupon, freq,T0,T1]
        x0 = 0
        bnds = ((0,1),)
        res = minimize(self.Z_Spread_finder, x0, method='SLSQP',args=args,bounds = bnds)
        return res.x[0]
    
    def Zero_volatility_Yield_Structure_caller(self,Benchmark_yield, Kappa, theta, xi,dt,freq,T0,T1,J):
        T = self.delta_computator(T0,T1)[1]
        day_forecast = int(round(T - self.Superflous(freq,T0,T1),1) * freq)+1
        H = Initialize_parameters().MCR(Benchmark_yield, Kappa, theta, xi,dt,day_forecast,J)[1:]
        return H

    def __init__(self,coupon,frequency,dayCount,price,Benchmark_yield,par,freq,T0,T1,Kappa, theta, xi,dt):
        self.volatools = self.Duration_computation(coupon,frequency,dayCount,price,T0,T1)
        self.Results = self.Bond_Pricer(coupon,frequency,dayCount,price,T0,T1)
        self.Bond_Price_Given_Yield = self.bond_price_Given_Yield_Curve(par,coupon, freq,T0,T1,Kappa, theta, xi,dt,Benchmark_yield)
        self.Superfl = self.Superflous(freq,T0,T1)
        self.delta = self.delta_computator(T0,T1)
        self.maturity_start = self.Maturity_start_calculator(T0,T1)
        self.zspread = self.Z_spread_Optimizor(coupon,frequency,dayCount,price,Benchmark_yield,T0,T1,par, freq,Kappa, theta, xi,dt)
        self.Yield_curve = self.Zero_volatility_Yield_Structure_caller(Benchmark_yield, Kappa, theta, xi,dt,freq,T0,T1,J)
        
class European_Style_option:
    
    def Callable_European_Bond_Price(self,Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date):
        a = Bond_function
        duration,bnd_price_change_minus1 = a.volatools[0],a.volatools[1]
        K_strike = Original_Price * (1 - bnd_price_change_minus1)
        b = Fisher_Black_Call(Original_Price,K_strike,Recall_date,Benchmark_yield,duration,Kappa,theta,xi,dt)
        Final_Price = Original_Price + b.price 
        return Final_Price
    
    def Option_Yield_Finder(self,Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date,coupon):
        Final_Price = self.Callable_European_Bond_Price(Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date)
        a = Bond_function
        z_spread = a.zspread
        OAS = self.OAS_finder(Bond_function,frequency,dayCount,Original_Price,Benchmark_yield,T0,T1,par, freq,Kappa, theta, xi,dt,Recall_date,coupon)
        Embedded_option = z_spread - OAS 
        return Embedded_option
        
    def Embedded_option_finder(self,oas,args):
        Real_Price,par, T, rate, coup , freq,T0,T1,Superflous = args
        freq = float(freq)
        periods = T*freq
        coupon = coup*par/freq
        #r = [rate[i] for i in range(int(periods))]
        dt = [((i+1)/freq) for i in range(int(periods))]
        Lst_price = []
        for j in range(len(rate.columns)):
            price = sum([coupon/(1+(rate.iloc[int(t*freq - 1),j]+oas)/freq)**(freq*(t-Superflous)) for t in dt]) + \
                par/(1+(rate.iloc[-1,j]+oas)/freq)**(freq*(T-Superflous))
            Lst_price.append(price)
        Lst_price = np.array(Lst_price).reshape(1,len(Lst_price))[0]
        return np.sum(((Lst_price-Real_Price)/Real_Price)**2)

    def OAS_finder(self,Bond_function,frequency,dayCount,Original_Price,Benchmark_yield,T0,T1,par, freq,Kappa, theta, xi,dt,Recall_date,coupon):
        a = Bond_function
        Delta_diff = a.Superfl
        T = a.delta[1]
        Real_Price  = self.Callable_European_Bond_Price(Bond_function,T0,T1,frequency,dayCount,Original_Price,Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date)
        rate = a.Yield_curve
        #rate = rate.values.reshape(1,len(rate))[0]
        args = [Real_Price,par, T, rate, coupon, freq,T0,T1,Delta_diff]
        x0 = 0
        bnds = ((0,1),)
        res = minimize(self.Embedded_option_finder, x0, method='SLSQP',args=args,bounds = bnds)
        return res.x[0]
    
    def __init__(self,Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date,coupon):
        self.OAS = self.OAS_finder(Bond_function,frequency,dayCount,Original_Price,Benchmark_yield,T0,T1,par, freq,Kappa, theta, xi,dt,Recall_date,coupon)
        self.OYF = self.Option_Yield_Finder(Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date,coupon)
        self.Final_Price = self.Callable_European_Bond_Price(Bond_function,T0,T1,frequency,dayCount,Original_Price,initial_Benchmark_yield,par,freq,Kappa, theta, xi,dt,Recall_date)
        
 if __name__ == '__main__':
  d0 = date.today()
  d1 = date(2026,8,15)
  delta = d1 - d0
  coupon = 0.05
  start = ql.Date().todaysDate()
  maturity = start + ql.Period(delta.days, ql.Days)
  frequency = ql.Semiannual
  dayCount = ql.Thirty360()
  price = 104
  par = 100
  T = round(delta.days/360,3)
  freq = 2
  Benchmark_yield = rfree.iloc[-1,0]
  dt = 1/2
  a = Bond_Evaluation(coupon,frequency,dayCount,price,Benchmark_yield,par,freq,d0,d1,Kappa,theta,xi,dt)
  Price_Given_Yield,Price,Z_spread,Duration,Bond_Price_Change_for_1_percent_interest_rate_higher = a.Bond_Price_Given_Yield,a.Results[0],a.zspread,a.volatools[0],a.volatools[1]
  print(Price_Given_Yield,Price,Z_spread,Duration,Bond_Price_Change_for_1_percent_interest_rate_higher)
  Original_Price = price
  Recall_date = 5 #Mid_long_term
  ESO = European_Style_option(a,d0,d1,frequency,dayCount,Original_Price,Benchmark_yield,par,freq,Kappa,theta,xi,dt,Recall_date,coupon)
  print(ESO.OAS,a.zspread,ESO.Final_Price)
        
