import pandas as pd


def main():
    p = 'data/raw/historical_rawdata.csv'
    df = pd.read_csv(p, low_memory=False)
    print('pm2 dtype', df['pm2'].dtype)
    print(df['pm2'].head(40))
    print('non-null count:', df['pm2'].notna().sum())


if __name__ == '__main__':
    main()
