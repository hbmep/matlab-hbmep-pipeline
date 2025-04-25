from pathlib import Path
import importlib.util
import importlib.metadata
import os
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.path.expanduser("~"))

args = SimpleNamespace(
    p_hbmep_config=ROOT / "auxf" / "internal" / "hbmep_config.toml",
    p_csv=HOME / 'matlab-hbmep-pipeline/proc/SUBID_V1_IMM_data/SUBID_V1_IMM_data_mepsize.csv',
    response=['RAPB', 'RFDI'],
    d_output=HOME / 'matlab-hbmep-pipeline' / 'testing',
)

p_hbmep_config = args.p_hbmep_config,
p_csv = args.p_csv,
response = args.response,
d_output = args.d_output

dnc_custom_args_path = Path(__file__).parent / 'dnc_custom_args.py'

if dnc_custom_args_path.exists():
    spec = importlib.util.spec_from_file_location("dnc_custom_args", dnc_custom_args_path)
    dnc_custom_args = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dnc_custom_args)

    # Overwrite args with values from dnc_custom_args if they exist
    for key in vars(args):
        if key in vars(dnc_custom_args.args):
            setattr(args, key, getattr(dnc_custom_args.args, key))

    print('Overwriting with dnc_custom_args.py conent.')


# Example dnc_custom_args.py in the same directory
# from hbmep_local.custom_args import args
# args.response = ['RECR', 'RFCR', 'RAPB', 'RADM', 'RFDI']