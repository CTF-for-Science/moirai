# Run with:
# python forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation
#  apptainer run --nv --cwd "/app/code" --bind "/mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai":"/app/code" /mmfs1/home/alexeyy/storage/CTF-for-Science/models/moirai/apptainer/gpu.sif python -u /app/code/ctf/forecast_ctf.py --dataset ODE_Lorenz --pair_id 1 --validation 0 --identifier lorenz_1

# ## Imports

import time
import torch
import pickle
import argparse
import numpy as np
import pprint as pp
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from ctf4science.data_module import load_validation_dataset, load_dataset, get_prediction_timesteps, get_validation_prediction_timesteps, get_validation_training_timesteps, get_metadata

top_dir = Path(__file__).parent.parent
pickle_dir = top_dir / 'pickles'
pickle_dir.mkdir(parents=True, exist_ok=True)

def main(args=None):
    # ## Model Parameters

    print("> Setting up model parameters")

    MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}
    SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
    #PDT = 20  # prediction length: any positive integer
    #CTX = 200  # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 32  # batch size: any positive integer
    #TEST = 100  # test set length: any positive integer

    # ## Data

    # Pair ids 2, 4: reconstruction
    # Pair ids 1, 3, 5-7: forecast
    # Pair ids 8, 9: burn-in
    pair_id = args.pair_id
    dataset = args.dataset
    validation = args.validation
    recon_ctx = args.recon_ctx # Context length for reconstruction

    print("> Setting up training data")

    md = get_metadata(dataset)

    if validation:
        train_data, val_data, init_data = load_validation_dataset(dataset, pair_id=pair_id)
        forecast_length = get_validation_prediction_timesteps(dataset, pair_id).shape[0]
    else:
        train_data, init_data = load_dataset(dataset, pair_id=pair_id)
        forecast_length = get_prediction_timesteps(dataset, pair_id).shape[0]

    print(f"> Predicting {dataset} for pair {pair_id} with forecast length {forecast_length}")

    delta_t = md['delta_t']

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        print(f"> Reconstruction task, using {recon_ctx} context length")
        train_mat = train_data[0]
        train_mat = train_mat[0:recon_ctx,:]
        forecast_length = forecast_length - recon_ctx
        df = pd.DataFrame(train_mat)
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        print(f"> Forecasting task, using {forecast_length} forecast length")
        train_mat = train_data[0]
        df = pd.DataFrame(train_mat)
    elif pair_id in [8, 9]:
        # Burn-in
        print(f"> Burn-in matrix of size {init_data.shape[0]}, using {forecast_length} forecast length")
        train_mat = init_data
        forecast_length = forecast_length - init_data.shape[0]
        df = pd.DataFrame(train_mat)
    else:
        raise ValueError(f"Pair id {pair_id} not supported")

    # Model variables
    PDT = 10
    CTX = min(df.shape[0], 200)
    TEST = 10

    print("> Model variables:")
    print(f"  PDT: {PDT}")
    print(f"  CTX: {CTX}")
    print(f"  TEST: {TEST}")

    # Loop until we have forecast_length data points
    forecast_loops = forecast_length // PDT + (forecast_length % PDT > 0)
    raw_preds = []
    for i in range(forecast_loops):
        print(f"> ({i+1}/{forecast_loops}) Input Shape:", df.shape)

        # Append TEST (forecast_length) of zeros to each column of the dataset
        df_test = pd.concat([df, pd.DataFrame(0.*np.ones((TEST, df.shape[1])), columns=df.columns)], axis=0)

        # Create DateTimeIndex starting from 0 seconds with delta_t intervals
        start_time = pd.Timestamp('2020-01-01')  
        timestamps = pd.date_range(start=start_time, 
                                    periods=len(df_test),
                                    freq=f'{delta_t}S')
        df_test.index = timestamps

        # Convert into GluonTS dataset with frequency in seconds
        ds = PandasDataset(dict(df_test), freq=f'{delta_t}S')

        # ## Run Model

        # Group time series into multivariate dataset
        grouper = MultivariateGrouper(len(ds))
        multivar_ds = grouper(ds)

        # Split into train/test set
        _, test_template = split(
            multivar_ds, offset=-TEST
        )  # assign last TEST time steps as test set

        # Construct rolling window evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=TEST // PDT,  # number of windows in rolling window evaluation
            distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
        )

        # Prepare model
        if MODEL == "moirai":
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=len(ds),
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
        elif MODEL == "moirai-moe":
            model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-small"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=16,
                num_samples=100,
                target_dim=len(ds),
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )

        # ## Forecast
        predictor = model.create_predictor(batch_size=BSZ, device='auto')
        forecasts = predictor.predict(test_data.input)

        # ## Create Prediction Matrix
        forecast_it = iter(forecasts)
        forecast = next(forecast_it)
        raw_pred = forecast.quantile(0.5)
        raw_preds.append(raw_pred)

        # ## Append to df
        df = pd.concat([df, pd.DataFrame(raw_pred, columns=df.columns)], axis=0)

        # Delete variables
        del model, predictor, forecasts, test_data, test_template, grouper, multivar_ds, forecast_it, forecast, raw_pred

    raw_pred = np.concatenate(raw_preds, axis=0)
    raw_pred = raw_pred[0:forecast_length, :]
    print("> Concatenated Shape", raw_pred.shape)

    print("> Creating prediction matrix")

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        pred = np.vstack([train_mat, raw_pred])
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        #pred = np.vstack([train_mat, raw_pred])
        pred = raw_pred
    elif pair_id in [8, 9]:
        # Burn-in
        pred = np.vstack([train_mat, raw_pred])
    else:
        raise ValueError(f"Pair id {pair_id} not supported") 

    print("> Predicted Matrix Shape:", pred.shape)
    
    if args.validation:
        print("> Expected Shape: ", val_data.shape)
    else:
        print("> Expected Shape: ", md['matrix_shapes'][f'X{pair_id}test.mat'])

    # ## Save prediction matrix
    with open(pickle_dir / f"{args.identifier}.pkl", "wb") as f:
        pickle.dump(pred, f)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, required=True, help="Identifier for the run")
    parser.add_argument('--dataset', type=str, default=None, required=True, help="Dataset to run (ODE_Lorenz or PDE_KS)")
    parser.add_argument('--pair_id', type=int, default=1, help="Pair_id to run (1-9)")
    parser.add_argument('--recon_ctx', type=int, default=20, help="Context length for reconstruction")
    parser.add_argument('--validation', type=int, default=0, help="Generate and use validation set")
    args = parser.parse_args()

    # Args
    print("> Args:")
    pp.pprint(vars(args), indent=2)

    # Start timing
    start_time = time.time()
    
    main(args)
    
    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Convert to HH:MM:SS format
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"> Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
