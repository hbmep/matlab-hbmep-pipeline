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
git clone https://github.com/hbmep/matlab-hbmep-pipeline.git
navigate to matlab-hbmep-pipeline/auxf/hbmep

# 2. Set up Python environment for hbMEP (asusming you have conda/miniconda installed)
conda create -n python-311 python=3.11 -y
conda activate python-311
python -m venv .venv
conda deactivate           # deactivate python-311 conda env
conda deactivate           # deactivate default conda env
pip install --upgrade pip
source .venv/bin/activate  # OR .venv\Scripts\activate.bat (if windows)

# 3. Install hbMEP locally
pip install .
```

## Usage

Once hbMEP is installed, you can call the main MATLAB wrapper:

```matlab
% Arguments:
%   loaderName    — name of your data-loading function (e.g. 'loader_bronxva' is an existing one)
%   dataDirectory — path to the folder containing your data (can be set to 'example-data' to get started)
%   muscleList    — cell array of muscles to analyze (reflecting a subset of names in config file of data)
analyse_ramp('loader_bronxva', 'example-data', ["RECR", "RFCR", "RAPB", "RADM", "RFDI"]);
```

This will:

1. Read EMG, trigger, intensity, and config files.  
2. Build a MATLAB table for each muscle.  
3. Fit the recruitment curve model via hbMEP.  
4. Return summary output (posterior distributions, diagnostic plots).


## License

Released under the MIT License. See [LICENSE](LICENSE) for full text.