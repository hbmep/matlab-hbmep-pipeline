import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from hbmep.model import BaseModel
from hbmep.model.standard import EPS
from hbmep import functional as F
from hbmep.util import site as site
import jax


class Config:
    def __init__(self, toml_path=None):
        self.toml_path = toml_path
        self.INTENSITY = "TMSInt"
        self.RESPONSE = []
        self.FEATURES = ["participant", "condition"]
        self.BUILD_DIR = "./"
        self.MCMC_PARAMS = {
            "num_chains": 4,
            "num_warmup": 400,
            "num_samples": 400,
            "thinning": 1,
        }
        self.NUTS = {
            "max_tree_depth": (10, 10),
            "target_accept_prob": 0.8,
        }
        self.MEP_DATA = {
            "mep_matrix_path": "",
            "mep_response": [],
            "mep_window": [-0.25, 0.25],
            "mep_size_window": [0.0065, 0.09],
        }


def get_ima_parameters(model):
    """Extract per-muscle IMA parameters (prob, mean, sd) aligned with model.response."""
    params = getattr(model, 'ima_params', {}) or {}
    probs = []
    means = []
    sds = []
    responses = getattr(model, 'response', [])
    for r in responses:
        p_match = None
        if isinstance(params, dict):
            if r in params:
                p_match = params[r]
            else:
                for k, v in params.items():
                    if str(k).strip().lower() == str(r).strip().lower():
                        p_match = v
                        break
        if isinstance(p_match, dict):
            p_val = float(p_match.get('prob', p_match.get('probability', 0.1)))
            m_val = float(p_match.get('mean', 0.1))
            s_val = float(p_match.get('sd', p_match.get('std', 1.0)))
        else:
            p_val = 0.1
            m_val = 0.1
            s_val = 1.0
        probs.append(p_val)
        means.append(m_val)
        sds.append(s_val)
    return jnp.array(probs), jnp.array(means), jnp.array(sds)


class rlpkpk(BaseModel):
    """Rectified Logistic model implemented locally with hierarchical indexing fixes.
    Supports both TMS and SCS stimulation types."""
    def __init__(self, *args, is_scs=False, stim_type=None, **kw):
        super(rlpkpk, self).__init__(*args, **kw)
        self.use_mixture = False
        self.use_ima = False
        self.ima_params = {}
        config = kw.get('config', {})
        if stim_type is not None:
            self.is_scs = (stim_type.lower() == 'scs')
        elif 'stim_type' in config:
            self.is_scs = (config['stim_type'].lower() == 'scs')
        else:
            self.is_scs = is_scs

    def load(self, df, **kwargs):
        if hasattr(super(), 'load'):
            df, encoder = super().load(df, **kwargs)
        else:
            df, encoder = super().process(df, **kwargs)
        self.encoder = encoder
        if not getattr(self, 'is_scs', False):
            if hasattr(self, 'features') and self.features:
                last_feat = self.features[-1]
                if last_feat in encoder:
                    classes = encoder[last_feat].classes_
                    if any(c.startswith('sc') for c in classes):
                        self.is_scs = True
        return df, encoder

    def process(self, df, **kwargs):
        return self.load(df, **kwargs)

    def rectified_logistic(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        # Calculate concrete feature counts to avoid TracerBoolConversionError
        if hasattr(self, 'encoder') and self.encoder is not None:
            num_features = [np.int64(len(self.encoder[f].classes_)) for f in self.features]
        else:
            num_features = np.max(features, axis=0) + 1

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = jnp.isfinite(response)

        # Hyper-priors (differentiated for SCS vs TMS)
        is_scs = getattr(self, 'is_scs', False)
        if is_scs:
            a_loc = pyro.sample(site.a.loc, dist.TruncatedNormal(5., 10., low=0))
            a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
        else:
            a_loc = pyro.sample(site.a.loc, dist.TruncatedNormal(50., 50., low=0))
            a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))  # changed this from 5.0 (potentially to accomodate SCS MEP sizes)
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))
        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Plate stack dimension order matches features columns: (F0, F1, ..., R)
        idx = tuple(features.T) + (slice(None),)

        # Priors
        with pyro.plate(site.num_response, self.num_response, dim=-1):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                a = pyro.sample(site.a, dist.TruncatedNormal(a_loc, a_scale, low=0))
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)
                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)
                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)
                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)
                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)
                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if getattr(self, 'use_ima', False):
            ima_probs, ima_means, ima_sds = get_ima_parameters(self)
            with pyro.plate(site.num_response, self.num_response, dim=-1):
                ima_const = pyro.sample(
                    "ima_const",
                    dist.TruncatedNormal(ima_means, ima_sds, low=0.)
                )

        if response is None:
            q = 0.
            p = jnp.zeros(self.num_response) if getattr(self, 'use_ima', False) else None
        else:
            if self.use_mixture:
                q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))
            if getattr(self, 'use_ima', False):
                p = ima_probs

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            mu = pyro.deterministic(
                site.mu,
                F.rectified_logistic(
                    intensity.reshape(-1, 1),
                    a[idx], b[idx], g[idx], h[idx], v[idx], EPS
                )
            )
            alpha, beta = self.gamma_likelihood(mu, c1[idx], c2[idx])

            if getattr(self, 'use_ima', False):
                mu_ima = pyro.deterministic("mu_ima", mu + ima_const)
                alpha_ima, beta_ima = self.gamma_likelihood(mu_ima, c1[idx], c2[idx])

            outlier_dist = dist.HalfNormal(scale=(g[idx] + h[idx]))

            if self.use_mixture and getattr(self, 'use_ima', False):
                w0 = (1.0 - q) * (1.0 - p)
                w1 = (1.0 - q) * p
                w2 = jnp.broadcast_to(q, w0.shape)
                probs = jnp.broadcast_to(jnp.stack([w0, w1, w2], axis=-1), (num_data, self.num_response, 3))
                obs_dist = dist.MixtureGeneral(
                    dist.Categorical(probs=probs),
                    [
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.Gamma(concentration=alpha_ima, rate=beta_ima),
                        outlier_dist
                    ]
                )
            elif getattr(self, 'use_ima', False):
                w0 = 1.0 - p
                w1 = p
                probs = jnp.broadcast_to(jnp.stack([w0, w1], axis=-1), (num_data, self.num_response, 2))
                obs_dist = dist.MixtureGeneral(
                    dist.Categorical(probs=probs),
                    [
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.Gamma(concentration=alpha_ima, rate=beta_ima)
                    ]
                )
            elif self.use_mixture:
                mixing_distribution = dist.Categorical(probs=jnp.stack([1 - q, q], axis=-1))
                component_distributions = [
                    dist.Gamma(concentration=alpha, rate=beta),
                    outlier_dist
                ]
                obs_dist = dist.MixtureGeneral(mixing_distribution, component_distributions)
            else:
                obs_dist = dist.Gamma(alpha, beta)

            with pyro.plate(site.num_response, self.num_response, dim=-1):
                with pyro.plate(site.num_data, num_data, dim=-2):
                    y_ = pyro.sample(
                        site.obs,
                        obs_dist,
                        obs=response
                    )
                    if self.use_mixture and not getattr(self, 'use_ima', False):
                        log_probs = obs_dist.component_log_probs(y_)
                        pyro.deterministic("p", log_probs - jax.nn.logsumexp(log_probs, axis=-1, keepdims=True))


class rlpkpkSmall(BaseModel):
    """Based on rl in base_agent.py (non-hierarchical priors).
    Supports both TMS and SCS stimulation types."""
    def __init__(self, *args, is_scs=False, stim_type=None, **kw):
        super(rlpkpkSmall, self).__init__(*args, **kw)
        self.use_mixture = False
        self.use_ima = False
        self.ima_params = {}
        config = kw.get('config', {})
        if stim_type is not None:
            self.is_scs = (stim_type.lower() == 'scs')
        elif 'stim_type' in config:
            self.is_scs = (config['stim_type'].lower() == 'scs')
        else:
            self.is_scs = is_scs

    def load(self, df, **kwargs):
        if hasattr(super(), 'load'):
            df, encoder = super().load(df, **kwargs)
        else:
            df, encoder = super().process(df, **kwargs)
        self.encoder = encoder
        if not getattr(self, 'is_scs', False):
            if hasattr(self, 'features') and self.features:
                last_feat = self.features[-1]
                if last_feat in encoder:
                    classes = encoder[last_feat].classes_
                    if any(c.startswith('sc') for c in classes):
                        self.is_scs = True
        return df, encoder

    def process(self, df, **kwargs):
        return self.load(df, **kwargs)

    def rectified_logistic(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        if hasattr(self, 'encoder') and self.encoder is not None:
            num_features = [np.int64(len(self.encoder[f].classes_)) for f in self.features]
        else:
            num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = jnp.isfinite(response)

        idx = tuple(features.T) + (slice(None),)

        # SCS vs TMS motor threshold prior parameterization
        is_scs = getattr(self, 'is_scs', False)
        if is_scs:
            a_loc = 5.0
            a_scale = 10.0
        else:
            a_loc = 50.0
            a_scale = 50.0

        with pyro.plate(site.num_response, self.num_response, dim=-1):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                a = pyro.sample(site.a, dist.TruncatedNormal(a_loc, a_scale, low=0.))
                b = pyro.sample(site.b, dist.HalfNormal(1.))
                g = pyro.sample(site.g, dist.HalfNormal(.1))
                h = pyro.sample(site.h, dist.HalfNormal(10.))  # changed this from 5.0 (potentially to accomodate SCS MEP sizes)
                v = pyro.sample(site.v, dist.HalfNormal(5.))
                c1 = pyro.sample(site.c1, dist.HalfNormal(5.))
                c2 = pyro.sample(site.c2, dist.HalfNormal(.5))

        if getattr(self, 'use_ima', False):
            ima_probs, ima_means, ima_sds = get_ima_parameters(self)
            with pyro.plate(site.num_response, self.num_response, dim=-1):
                ima_const = pyro.sample(
                    "ima_const",
                    dist.TruncatedNormal(ima_means, ima_sds, low=0.)
                )

        if response is None:
            q = 0.
            p = jnp.zeros(self.num_response) if getattr(self, 'use_ima', False) else None
        else:
            if self.use_mixture:
                q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))
            if getattr(self, 'use_ima', False):
                p = ima_probs

        with pyro.handlers.mask(mask=mask_obs):
            mu = pyro.deterministic(
                site.mu,
                F.rectified_logistic(
                    intensity.reshape(-1, 1),
                    a[idx], b[idx], g[idx], h[idx], v[idx], EPS
                )
            )
            alpha, beta = self.gamma_likelihood(mu, c1[idx], c2[idx])

            if getattr(self, 'use_ima', False):
                mu_ima = pyro.deterministic("mu_ima", mu + ima_const)
                alpha_ima, beta_ima = self.gamma_likelihood(mu_ima, c1[idx], c2[idx])

            outlier_dist = dist.HalfNormal(scale=10.0)

            if self.use_mixture and getattr(self, 'use_ima', False):
                w0 = (1.0 - q) * (1.0 - p)
                w1 = (1.0 - q) * p
                w2 = jnp.broadcast_to(q, w0.shape)
                probs = jnp.broadcast_to(jnp.stack([w0, w1, w2], axis=-1), (num_data, self.num_response, 3))
                obs_dist = dist.MixtureGeneral(
                    dist.Categorical(probs=probs),
                    [
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.Gamma(concentration=alpha_ima, rate=beta_ima),
                        outlier_dist
                    ]
                )
            elif getattr(self, 'use_ima', False):
                w0 = 1.0 - p
                w1 = p
                probs = jnp.broadcast_to(jnp.stack([w0, w1], axis=-1), (num_data, self.num_response, 2))
                obs_dist = dist.MixtureGeneral(
                    dist.Categorical(probs=probs),
                    [
                        dist.Gamma(concentration=alpha, rate=beta),
                        dist.Gamma(concentration=alpha_ima, rate=beta_ima)
                    ]
                )
            elif self.use_mixture:
                mixing_distribution = dist.Categorical(probs=jnp.stack([1 - q, q], axis=-1))
                component_distributions = [
                    dist.Gamma(concentration=alpha, rate=beta),
                    outlier_dist
                ]
                obs_dist = dist.MixtureGeneral(mixing_distribution, component_distributions)
            else:
                obs_dist = dist.Gamma(alpha, beta)

            with pyro.plate(site.num_response, self.num_response, dim=-1):
                with pyro.plate(site.num_data, num_data, dim=-2):
                    pyro.sample(
                        site.obs,
                        obs_dist,
                        obs=response
                    )


class rlaucSmall(rlpkpkSmall):
    """Rectified Logistic model for AUC MEP size (uVs).
    Currently mirrors rlpkpkSmall with identical priors and rectified logistic functional."""
    pass


class rlauc(rlpkpk):
    """Rectified Logistic hierarchical model for AUC MEP size (uVs)."""
    pass


# Backward compatibility aliases
rlTMSpkpkSmall = rlpkpkSmall
rlTMSpkpk = rlpkpk
