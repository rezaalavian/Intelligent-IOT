from pathlib import Path
import torch
from models.model_io import load_model
from models.spatiotemporal.train import load_training_frame, build_windows, chronological_split


def main(artifact: str | Path = None):
    frame = load_training_frame('data/raw/historical_rawdata.csv')
    X, y, cols = build_windows(frame)
    split = chronological_split(X, y)
    import numpy as np
    print('X_train NaN count:', int(np.isnan(split.X_train).sum()))
    print('X_val NaN count:', int(np.isnan(split.X_val).sum()))
    print('X_test NaN count:', int(np.isnan(split.X_test).sum()))

    if artifact is None:
        # find latest artifact in saved_models
        folder = Path('models/saved_models')
        files = sorted(folder.glob('spatiotemporal_model_*.pt'))
        if not files:
            print('No artifact found')
            return
        artifact = files[-1]

    print('Loading model:', artifact)
    model = load_model(artifact)
    model.eval()

    # inspect normalization stats
    try:
        print('input_mean finite:', bool(torch.isfinite(model.input_mean).all()))
        print('input_std finite:', bool(torch.isfinite(model.input_std).all()))
        print('input_std min:', float(model.input_std.min()))
    except Exception:
        print('model has no input_mean/std buffers')

    import numpy as np

    with torch.no_grad():
        x_train = torch.tensor(split.X_train, dtype=torch.float32)
        x_val = torch.tensor(split.X_val, dtype=torch.float32)
        x_test = torch.tensor(split.X_test, dtype=torch.float32)
        train_pred = model(x_train).detach().cpu().numpy()
        val_pred = model(x_val).detach().cpu().numpy()
        test_pred = model(x_test).detach().cpu().numpy()

    print('train_pred NaN count:', int(np.isnan(train_pred).sum()))
    print('val_pred NaN count:', int(np.isnan(val_pred).sum()))
    print('test_pred NaN count:', int(np.isnan(test_pred).sum()))


if __name__ == '__main__':
    main()
