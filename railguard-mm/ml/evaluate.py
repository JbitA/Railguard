from __future__ import annotations
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    p=argparse.ArgumentParser(); p.add_argument("csv")
    a=p.parse_args(); df=pd.read_csv(a.csv).dropna(subset=["measured","predicted"])
    print("MAE",mean_absolute_error(df.measured,df.predicted))
    print("RMSE",mean_squared_error(df.measured,df.predicted)**0.5)
if __name__=="__main__": main()
