import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import arviz as az
from hbmep.config import Config
from hbmep_local.model import RectifiedLogistic  # from hbmep.model.tms import RectifiedLogistic
from hbmep.model.utils import Site as site


def main(p_hbmep_config, p_csv, response, d_output):
    # Load hbMEP configuration
    cfg_hbmep = Config(toml_path=p_hbmep_config)
    dfo = pd.read_csv(str(p_csv))

    # Clean and preprocess data
    dfo[response] = dfo[response].apply(
        lambda col: col.map(lambda x: np.nan if x is not None and x <= 0 else x)
    )
    # Currently losing entire participants because of this line:
    # rows_to_drop = np.any(dfo[response].isna(), axis=1)
    # dfo = dfo[~rows_to_drop]
    # dfo = dfo.reset_index(drop=True).sort_index()

    # Stim mode
    stim_type = dfo.loc[0, "condition"]
    stim_mode = stim_type.split('_')[0]

    # Update hbMEP configuration
    cfg_hbmep.INTENSITY = f'intensity'
    cfg_hbmep.RESPONSE = response
    cfg_hbmep.FEATURES = ["participant", "condition"]
    cfg_hbmep.BUILD_DIR = d_output

    # Initialize model
    model = RectifiedLogistic(config=cfg_hbmep)
    model.use_mixture = True
    model.smooth = True
    model.s50parameterization = False

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

    # # Create and predict using the model
    # prediction_df = model.make_prediction_dataset(df=df, num_points=100)
    # posterior_predictive = model.predict(
    #     df=prediction_df, posterior_samples=posterior_samples
    # )
    #
    # model.render_recruitment_curves(
    #     df=df,
    #     encoder_dict=encoder_dict,
    #     posterior_samples=posterior_samples,
    #     prediction_df=prediction_df,
    #     posterior_predictive=posterior_predictive
    # )
    #
    # model.print_summary(samples=posterior_samples)
    #


    if site.outlier_prob in posterior_samples.keys():
        posterior_samples[site.outlier_prob] = posterior_samples[site.outlier_prob] * 0

    prediction_df = model.make_prediction_dataset(df=df, num_points=100)
    posterior_predictive = model.predict(
        df=prediction_df, posterior_samples=posterior_samples
    )

    # Optionally render curves
    # model.mep_window = [-0.25, 0.25]
    # model.mep_size_window = [0.005, 0.09]
    # model.mep_response = response  # if you introduce mep you need to address this
    print(f'Rendering to: {model.build_dir}')
    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive,
        #  mep_matrix=mep_stype
    )

    model.render_predictive_check(
        df=df,
        encoder_dict=encoder_dict,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )

    inference_data = az.from_numpyro(mcmc)

    print('Generating and saving HDI summary...')
    summary = az.summary(inference_data, hdi_prob=0.95)
    summary.to_csv(Path(model.build_dir) / 'summary.csv')
    print('Done.')