import os
import pickle
import argparse
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from hbmep.config import Config
from hbmep.model.tms import RectifiedLogistic
from hbmep.model.utils import Site as site
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main(p_hbmep_config, p_csv, response, d_output):
    # Load hbMEP configuration
    cfg_hbmep = Config(toml_path=p_hbmep_config)
    dfo = pd.read_csv(p_csv)

    # Clean and preprocess data
    dfo[response] = dfo[response].applymap(
        lambda x: np.nan if x is not None and x <= 0 else x
    )
    # Currently losing entire participants because of this line:
    rows_to_drop = np.any(dfo[response].isna(), axis=1)
    dfo = dfo[~rows_to_drop]
    dfo = dfo.reset_index(drop=True).sort_index()

    # Stim mode
    stim_type = dfo.loc[0, "condition"]
    stim_mode = stim_type.split('_')[0]

    # Update hbMEP configuration
    cfg_hbmep.INTENSITY = f'intensity'
    cfg_hbmep.RESPONSE = response
    cfg_hbmep.FEATURES = ["participant"]
    cfg_hbmep.BUILD_DIR = d_output

    # Initialize model
    model = RectifiedLogistic(config=cfg_hbmep)

    # Load data into the model
    df, encoder_dict = model.load(df=dfo)

    inference_path = Path(cfg_hbmep.BUILD_DIR) / 'inference.pkl'
    if inference_path.exists():
        with open(inference_path, "rb") as f:
            model, mcmc, posterior_samples = pickle.load(f)
    else:
        model.plot(df=df, encoder_dict=encoder_dict)

        mcmc, posterior_samples = model.run_inference(df=df)

        # Save the model and inference results
        with open(inference_path, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples), f)

    # Create and predict using the model
    prediction_df = model.make_prediction_dataset(df=df, num_points=100)
    posterior_predictive = model.predict(
        df=prediction_df, posterior_samples=posterior_samples
    )

    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )

    model.print_summary(samples=posterior_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hbMEP recruitment curve model.")
    parser.add_argument("--p_hbmep_config", required=True, help="Path to the hbMEP configuration file.")
    parser.add_argument("--p_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--response", nargs="+", required=True, help="Response column(s) in the CSV.")
    parser.add_argument("--d_output", required=True, help="Output directory for model results.")
    
    args = parser.parse_args()

    main(
        p_hbmep_config=args.p_hbmep_config,
        p_csv=args.p_csv,
        response=args.response,
        d_output=args.d_output
    )
