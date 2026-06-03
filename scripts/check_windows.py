from models.spatiotemporal.train import load_training_frame, build_windows


def main():
    frame = load_training_frame('data/raw/historical_rawdata.csv')
    print('frame_rows=', len(frame))
    X, y, cols = build_windows(frame)
    print('X.shape=', X.shape)
    print('y.shape=', y.shape)
    print('feature_count=', len(cols))


if __name__ == '__main__':
    main()
