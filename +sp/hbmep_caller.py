import argparse
import sys
import io
import run_model
from hbmep_local import custom_args

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided externally - this is used for debugging.")
        args = custom_args.args

    else:
        parser = argparse.ArgumentParser(description="Run hbMEP recruitment curve model.")
        parser.add_argument("--p_hbmep_config", required=True, help="Path to the hbMEP configuration file.")
        parser.add_argument("--p_csv", required=True, help="Path to the input CSV file.")
        parser.add_argument("--response", nargs="+", required=True, help="Response column(s) in the CSV.")
        parser.add_argument("--d_output", required=True, help="Output directory for model results.")
        parser.add_argument("--units_intensity", default="A. U.", help="Units of intensity.")
        parser.add_argument("--units_mepsize", default="A. U.", help="Units of MEP size.")
        parser.add_argument("--p_postproc", default="", help="Path to post-processing module.")

        args = parser.parse_args()

    postprocessing_helper = {}
    postprocessing_helper['units_mepsize'] = args.units_mepsize
    postprocessing_helper['units_intensity'] = args.units_intensity
    postprocessing_helper['d_output'] = args.d_output

    run_model.main(
        p_hbmep_config=args.p_hbmep_config,
        p_csv=args.p_csv,
        response=args.response,
        d_output=args.d_output,
        p_postproc=args.p_postproc,
        postprocessing_helper=postprocessing_helper,
    )
