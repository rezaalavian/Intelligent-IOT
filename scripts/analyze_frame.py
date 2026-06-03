from models.spatiotemporal.train import load_training_frame


def main():
    frame = load_training_frame('data/raw/historical_rawdata.csv')
    num_cols = frame.select_dtypes(include=['number']).columns.tolist()
    print('Numeric columns count:', len(num_cols))
    for col in num_cols[:40]:
        nans = frame[col].isna().sum()
        print(f"{col}: nans={nans}")


if __name__ == '__main__':
    main()
