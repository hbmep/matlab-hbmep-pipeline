import argparse
import sys
import io
import os

# JAX/NumPyro Windows x64 stability configuration
os.environ["JAX_ENABLE_X64"] = "True"
if os.name == 'nt':
    # Force 1 device to avoid sharding bugs on Windows x64
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
else:
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={cpu_count}")

import time
import run_model
from hbmep_local import custom_args


import atexit


class MatlabProgressStream:
    """Wraps output stream so progress bars that emit carriage returns (\r)
    are formatted as clean vertical milestone updates with newlines (\n).
    This eliminates horizontal runaway scrolling in MATLAB's -nodesktop window."""
    def __init__(self, target, min_interval=2.0):
        self.target = target
        self.min_interval = min_interval
        self.last_write = 0.0

    def write(self, s):
        if not s:
            return

        if '\r' in s:
            clean_s = s.replace('\r', '').strip('\r\n')
            if not clean_s:
                return

            lines = [l for l in clean_s.split('\n') if l.strip()]
            if not lines:
                return
            clean_s = lines[-1].strip()

            now = time.time()
            is_done = ('100%' in clean_s)

            if is_done or (now - self.last_write >= self.min_interval):
                self.target.write(clean_s + '\n')
                self.target.flush()
                self.last_write = now
        else:
            self.target.write(s)
            self.target.flush()

    def flush(self):
        self.target.flush()


sys.stdout = MatlabProgressStream(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
sys.stderr = MatlabProgressStream(io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace'))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided externally - this is used for debugging.")
        args = custom_args.args
        mcmc_params = None

    else:
        parser = argparse.ArgumentParser(description="Run hbMEP recruitment curve model.")
        # parser.add_argument("--p_hbmep_config", default=None, help="Path to the hbMEP configuration file (optional).")
        parser.add_argument("--p_csv", required=True, help="Path to the input CSV file.")
        parser.add_argument("--response", nargs="+", required=True, help="Response column(s) in the CSV.")
        parser.add_argument("--d_output", required=True, help="Output directory for model results.")
        parser.add_argument("--units_intensity", default="A. U.", help="Units of intensity.")
        parser.add_argument("--units_mepsize", default="A. U.", help="Units of MEP size.")
        parser.add_argument("--p_postproc", default="", help="Path to post-processing module.")
        parser.add_argument("--model", default="rlpkpkSmall", help="Model class to use.")
        parser.add_argument("--stim_type", default=None, help="Stimulation type: 'scs' or 'tms'.")
        parser.add_argument("--use_mixture", action="store_true", help="Use mixture model for outliers.")
        parser.add_argument("--use_ima", action="store_true", help="Use Involuntary Motor Activity (IMA) model.")
        parser.add_argument("--p_ima_params", default="", help="Path to JSON file with IMA parameters.")
        parser.add_argument("--ima_params", default="", help="Inline JSON string with IMA parameters.")
        parser.add_argument("--num_warmup", type=int, default=400, help="Number of warmup iterations.")
        parser.add_argument("--num_samples", type=int, default=400, help="Number of samples.")
        parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
        parser.add_argument("--thinning", type=int, default=1, help="Thinning factor.")
        parser.add_argument("--max_tree_depth", type=int, default=10, help="Maximum tree depth for NUTS.")
        parser.add_argument("--target_accept_prob", type=float, default=0.8, help="Target acceptance probability for NUTS.")

        args = parser.parse_args()
        mcmc_params = {
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "thinning": args.thinning,
        }
        nuts_params = {
            "max_tree_depth": (args.max_tree_depth, args.max_tree_depth),
            "target_accept_prob": args.target_accept_prob,
        }

    import json
    ima_params = {}
    if getattr(args, 'p_ima_params', None) and os.path.isfile(args.p_ima_params):
        try:
            with open(args.p_ima_params, 'r') as f:
                ima_params = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load IMA parameters from {args.p_ima_params}: {e}")
    elif getattr(args, 'ima_params', None):
        try:
            ima_params = json.loads(args.ima_params)
        except Exception as e:
            print(f"Warning: Failed to parse IMA parameters JSON: {e}")

    postprocessing_helper = {}
    postprocessing_helper['units_mepsize'] = args.units_mepsize
    postprocessing_helper['units_intensity'] = args.units_intensity
    postprocessing_helper['d_output'] = args.d_output
    postprocessing_helper['stim_type'] = getattr(args, 'stim_type', None)

    run_model.main(
        p_csv=args.p_csv,
        response=args.response,
        d_output=args.d_output,
        p_postproc=args.p_postproc,
        postprocessing_helper=postprocessing_helper,
        model_name=args.model,
        stim_type=getattr(args, 'stim_type', None),
        use_mixture=args.use_mixture,
        use_ima=getattr(args, 'use_ima', False),
        ima_params=ima_params,
        mcmc_params=mcmc_params,
        nuts_params=nuts_params,
        #p_hbmep_config=args.p_hbmep_config,
    )
