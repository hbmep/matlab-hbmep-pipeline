import os

import numpyro
numpyro.set_host_device_count(1 if os.name == "nt" else 4)

import hbmep
from hbmep.util import site as site
import hbmep_local.model as local_models
import pickle
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
import arviz as az


def main(p_csv, response, d_output, p_postproc, postprocessing_helper, p_hbmep_config=None, model_name="rlTMSpkpkSmall", use_mixture=False):
    # Load hbMEP configuration
    cfg_hbmep = local_models.Config(toml_path=p_hbmep_config)
    dfo = pd.read_csv(str(p_csv))

    # Clean and preprocess data
    dfo[response] = dfo[response].apply(
        lambda col: col.map(lambda x: np.nan if x is not None and x <= 0 else x)
    ).astype(np.float64)

    # Update hbMEP configuration
    cfg_hbmep.INTENSITY = f'intensity'
    dfo[cfg_hbmep.INTENSITY] = dfo[cfg_hbmep.INTENSITY].astype(np.float64)
    cfg_hbmep.RESPONSE = response
    cfg_hbmep.FEATURES = ["participant", "condition"]
    cfg_hbmep.BUILD_DIR = d_output
    
    # Ensure output directory exists
    Path(d_output).mkdir(parents=True, exist_ok=True)

    # Initialize model
    base_config = {
        "variables": {
            "intensity": cfg_hbmep.INTENSITY,
            "features": cfg_hbmep.FEATURES,
            "response": cfg_hbmep.RESPONSE,
        },
        "mcmc": cfg_hbmep.MCMC_PARAMS,
        "mep_data": cfg_hbmep.MEP_DATA,
    }
    
    model_class = getattr(local_models, model_name)
    model = model_class(config=base_config)
    
    # Set mixture mode
    model.use_mixture = use_mixture
    
    model._model = model.rectified_logistic
    model.build_dir = d_output

    # Load data into the model
    df, encoder_dict = model.load(df=dfo)
    for f in model.features:
        df[f] = df[f].astype(np.int64)

    inference_path = Path(cfg_hbmep.BUILD_DIR) / 'inference.pkl'
    if inference_path.exists():
        with open(inference_path, "rb") as f:
            model, mcmc, posterior_samples = pickle.load(f)
        
        # Patch for older pickle files where attributes were named differently
        if 'response' in model.__dict__ and not hasattr(model, '_response'):
            model._response = model.__dict__.pop('response')
        if not hasattr(model, '_num_response'):
            model._num_response = len(model.response) if hasattr(model, 'response') else None
        
        # Restore encoder from the load call above
        model.encoder = encoder_dict

        # Patch for older pickle files where 'key' was 'rng_key'
        if 'rng_key' in model.__dict__ and not hasattr(model, 'key'):
            model.key = model.__dict__['rng_key']
        
        # Patch for features
        if 'features' in model.__dict__ and not hasattr(model, '_features'):
            model._features = model.__dict__.get('features', [])
        
        # Update build_dir to the current output directory (it might have been a Windows path)
        model.build_dir = d_output

        # Get chained samples for ArviZ summary
        posterior_samples_chained = mcmc.get_samples(group_by_chain=True)
    else:
        model.plot(df=df, encoder_dict=encoder_dict)

        mcmc, posterior_samples = model.run(df=df)

        # Get samples with chain dimension for ArviZ summary
        posterior_samples_chained = mcmc.get_samples(group_by_chain=True)

        # Save the model and inference results
        with open(inference_path, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples), f)

    if site.outlier_prob in posterior_samples.keys():
        posterior_samples[site.outlier_prob] = posterior_samples[site.outlier_prob] * 0

    prediction_df = model.make_prediction_dataset(df=df, num_points=100)
    posterior_predictive = model.predict(
        df=prediction_df, posterior=posterior_samples
    )

    print(f'Rendering to: {model.build_dir}')
    model.plot_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior=posterior_samples,
        prediction_df=prediction_df,
        predictive=posterior_predictive,
        prediction_prob=0.95
    )

    if hasattr(model, "plot_predictive"):
        model.plot_predictive(
            df=df,
            encoder_dict=encoder_dict,
            prediction_df=prediction_df,
            predictive=posterior_predictive
        )

    print('Generating and saving HDI summary...')
    summary = az.summary(posterior_samples_chained, hdi_prob=0.95)
    summary.to_csv(Path(model.build_dir) / 'summary.csv')
    
    import scipy.io
    sanitized_posterior = {}
    for k, v in posterior_samples.items():
        new_key = k.replace('α', 'alpha').replace('β', 'beta').replace('μ', 'mu').replace('µ', 'mu')
        new_key = new_key.replace('c₁', 'c1').replace('c₂', 'c2').replace('ℓ', 'ell')
        new_key = new_key.replace('₁', '1').replace('₂', '2').replace('₃', '3').replace('₄', '4')
        sanitized_posterior[new_key] = v
    scipy.io.savemat(Path(model.build_dir) / 'posterior.mat', sanitized_posterior)
    
    print('Done.')

    if p_postproc:
        p_postproc = Path(p_postproc)
        if not p_postproc.is_file():
            raise FileNotFoundError(f"Post-processing script not found: {p_postproc}")
        # load the module from an arbitrary file path
        spec = importlib.util.spec_from_file_location("postproc_module", str(p_postproc))
        postproc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(postproc_module)
        postproc_module.postprocess(
            model=model,
            df=df,
            encoder_dict=encoder_dict,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive,
            postprocessing_helper=postprocessing_helper,
        )
