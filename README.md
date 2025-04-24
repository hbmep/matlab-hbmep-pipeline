# matlab-hbmep-pipeline

MATLAB pre-processing and interface to **hbMEP**, a hierarchical Bayesian package for estimating MEP recruitment curves.

## Features

1. **Data loading**  
   - Raw EMG and trigger `.csv` files  
   - Stimulation-intensity `.csv` file  
   - Small configuration file  

2. **Data preparation**  
   Converts inputs into a MATLAB table suitable for hbMEP.

3. **Model execution**  
   Calls hbMEP on the prepared table to fit recruitment curves.

## Installation

```bash
# 1. Clone this repository
git clone --recurse-submodules https://github.com/hbmep/matlab-hbmep-pipeline.git
navigate to matlab-hbmep-pipeline/auxf/hbmep

# 2. Set up Python environment for hbMEP (asusming you have conda/miniconda installed)
conda create -n python-311 python=3.11 -y
conda activate python-311
python -m venv .venv
conda deactivate
conda deactivate
.venv\Scripts\activate.bat
(replace the line above with source .venv/bin/activate if on linux)

# 3. Install hbMEP locally
pip install .
```

## Usage

Once hbMEP is installed, you can call the main MATLAB wrapper:
1. Open up matlab and navigate to the root matlab-hbmep-pipeline directory 
2. analyse_ramp('loader_bronxva', 'example-data', ["RECR", "RFCR", "RAPB", "RADM", "RFDI"]);

```matlab
% Arguments:
%   h_loader    — name of your data-loading function in loaders directory (e.g. 'loader_bronxva' is an existing one)
%   p_data — path to the main data.csv
%   response    — string array of muscles to analyze (reflecting a subset of names in config file of data)
```

This will:

1. Read EMG, trigger, intensity, and config files.  
2. Build a MATLAB table for each muscle.  
3. Fit the recruitment curve model via hbMEP.  
4. Return summary output (posterior distributions, diagnostic plots).


## License

Released under the MIT License. See [LICENSE](LICENSE) for full text.