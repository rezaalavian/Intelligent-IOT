from models.spatiotemporal.train import train


def main():
    print('Quick train: max_rows=2000, target=o3')
    result = train(path='data/raw/historical_rawdata.csv', output_path='models/saved_models/quick_smoke.pt', max_rows=2000, target_column='o3')
    print(result)


if __name__ == '__main__':
    main()
