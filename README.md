# ngiab-cal

A Python CLI tool to simplify hydrologic model calibration for NextGen In A Box (NGIAB) workflows.

## Table of Contents
- [What is this?](#what-is-this)
- [Installation](#installation)
- [Requirements](#requirements)
- [Basic Usage](#basic-usage)
- [Advanced Options](#advanced-options)
- [Calibration Process](#calibration-process)
- [Calibration Configuration File](#calibration-configuration-file)
- [Example: Calibrating CAMELS Basins](#example-calibrating-camels-basins)
- [How It Works](#how-it-works)
- [How is ngen-cal running?](#how-is-ngen-cal-running)
- [Development](#development)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## What is this?

ngiab-cal is a utility that works with the [NGIAB folder structure](https://docs.ciroh.org/training-NGIAB-101/data-preparation.html#nextgen-run-directory-structure-ngen-run). It automates the creation of a calibration directory with all necessary configuration files to run a modified version of [ngen-cal](https://github.com/CIROH-UA/ngen-cal/tree/ngiab_cal).

The tool simplifies these key tasks:
- Creating calibration configurations
- Running the calibration process using Docker
- Copying calibrated parameters back to your model configuration

## Installation

Several installation options are available:

### Using `uvx` (recommended: for its speed,and efficient environment management)
```bash
# Run directly without installation
uvx ngiab-cal --help

# Or install as a tool
uv tool install ngiab-cal
ngiab-cal --help
```

### Using `pipx`
```bash
pipx install ngiab-cal
ngiab-cal --help
```

### Traditional pip installation
```bash
pip install ngiab-cal
python -m ngiab_cal --help
```

## Requirements

- Python 3.10+
- Docker (for running calibrations)
- Internet connection (for downloading USGS data)

## Basic Usage

```bash
# Create calibration configuration
ngiab-cal /path/to/ngiab/data/folder -g USGS_GAGE_ID

# Create and run calibration (200 iterations)
ngiab-cal /path/to/ngiab/data/folder -g USGS_GAGE_ID --run -i 200

# Force recreation of calibration configuration
ngiab-cal /path/to/ngiab/data/folder -g USGS_GAGE_ID -f
```

## Advanced Options

```
usage: ngiab-cal [-h] [-g GAGE] [-f] [--run] [-i ITERATIONS] [--debug] [-w WARMUP] [--calibration_ratio CALIBRATION_RATIO]
                  data_folder

Create a calibration config for ngen-cal

positional arguments:
  data_folder           Path to the folder you wish to calibrate

options:
  -h, --help            show this help message and exit
  -g GAGE, --gage GAGE  Gage ID to use for calibration
  -f, --force           Overwrite existing configuration
  --run                 Try to automatically run the calibration, this may be unstable
  -i ITERATIONS, --iterations ITERATIONS
                        Default:100 number of iterations to calibrate for

  --debug               enable debug logging
  -w WARMUP, --warmup WARMUP
                        Default:365
                        Number of days at the beginning of the simulation
                         to exclude from calibration objective metric calculation
  --calibration_ratio CALIBRATION_RATIO, --cr CALIBRATION_RATIO
                        Default:0.5
                        How to split time after warmup into calibration and validation.
                        1 == 100% calibration, 0 == 100% validation, 0.8 == 80% calibration 20% validation
```

## Calibration Process

The tool applies a standard hydrological modeling workflow, which involves warmup, calibration, and validation periods. The --warmup period is crucial for allowing the model to reach a stable state before its performance is evaluated against observed data. Following the warmup, the remaining period is typically divided into calibration (where model parameters are adjusted to match observations) and validation (where the model's performance with the calibrated parameters is tested on an independent dataset). The tool facilitates this split, as detailed in the diagram and options below. 

```
Default calibration settings on a 5 year period
|   year 1   |   year 2   |   year 3   |   year 4   |   year 5   |
|<- warmup ->|<-     calibration     ->|                         |
|<-             warmup               ->|<-      validation     ->|
```

## Example: Calibrating CAMELS Basins

Here's a script to calibrate all CAMELS basins:

```bash
#!/bin/bash
# Download a list of CAMELS gage ids
wget https://raw.githubusercontent.com/peckhams/nextgen_basin_repo/5e1317256a9365ae3a24a250358314e1e9ffc339/CAMELS/Data/camels_name.txt ./camels_name.txt
output_folder=$(cat ~/.ngiab/preprocessor)
while read line
do
    gage=$(echo "$line" | cut -d ';' -f 1)
    echo $gage
    # subset the hydrofabric, calculate mean-average area forcings, generate model config files
    uvx --from ngiab_data_preprocess cli -i gage-"$gage" -sfr --start 2007-10-01 --end 2013-09-30
    # calibrate gage for 200 iterations
    uvx ngiab-cal "$output_folder"/gage-"$gage" -g "$gage" --run -i 200
done < <(tail -n +2 ./camels_name.txt)
```

## Calibration Configuration File

ngiab-cal generates a configuration file at `calibration/ngen_cal_conf.yaml` that controls the calibration process. This file is created from a template with the following key sections:

### General Configuration
```yaml
general:
  strategy:
    type: estimation
    algorithm: dds         # Uses Dynamically Dimensioned Search algorithm
  name: calib              # Don't modify this
  log: true               # Enable logging
  workdir: /ngen/ngen/data/calibration  # Don't modify this working directory in the Docker container
  yaml_file: /ngen/ngen/data/calibration/ngen_cal_conf.yaml # Don't modify this either
  iterations: 100         # Number of calibration iterations (customizable with -i flag)
  restart: 0              # Start from beginning (0) or resume from iteration
```

### Model Parameters
The file defines parameters for both CFE (Conceptual Functional Equivalent) and Noah-OWP-Modular (an extended, refactored version of the Noah-MP land surface model) hydrologic models:

```yaml
CFE:
  - name: b               # CFE parameter name
    min: 2.0              # Minimum allowed value
    max: 15.0             # Maximum allowed value
    init: 4.05            # Initial value
  - name: satpsi
    min: 0.03
    max: 0.955
    init: 0.355
  # Additional parameters...
```

For each parameter, the configuration specifies:
- `name`: Parameter identifier
- `min`: Minimum allowed value during calibration
- `max`: Maximum allowed value during calibration
- `init`: Initial value

### Evaluation Configuration
```yaml
eval_params:
  objective: kge           # Kling-Gupta Efficiency as objective function
  evaluation_start: "..."  # Start time for calibration period
  evaluation_stop: "..."   # End time for calibration period
  valid_start_time: "..."  # Start time including warmup
  valid_end_time: "..."    # End time of simulation
  # Additional time parameters...
  basinID: 01646500        # USGS gage ID
  site_name: "USGS 01646500: "  # Label for plots
```

These time periods follow the calibration process diagram shown earlier, with separate periods for warmup, calibration, and validation.

## How It Works

1. **Input Validation** - Checks that all required files are present
2. **USGS Data Download** - Retrieves observed streamflow for calibration
3. **Configuration Creation** - Generates necessary files for ngen-cal
4. **Docker Execution** - Runs the calibration in a containerized environment
5. **Parameter Extraction** - Copies calibrated parameters back to your configuration

## How is ngen-cal running?

The tool uses a custom [branch of CIROH-UA/ngen-cal](https://github.com/CIROH-UA/ngen-cal/tree/ngiab_cal) called `ngiab_cal`. This branch contains:
- A modified version of ngen-cal
- A Dockerfile that builds on top of the Next Gen In-A-Box container
- Installation of ngen-cal inside the container

When you provide the `--run` argument, it downloads a pre-built Docker image (`joshcu/ngiab-cal:demo`) to run the calibration.

## Limitations

ngiab-cal has several important limitations to be aware of:

1. **Limited Calibration Algorithms**: Currently only configured to use the Dynamically Dimensioned Search (DDS) algorithm. Other algorithms available in the main ngen-cal branch are not supported.

2. **Limited Model Support**: Only calibrates parameters for CFE (Conceptual Functional Equivalent) and NoahOWP-Modular hydrologic models. Other models in the NextGen framework are not currently supported.

3. **Single-Gage Calibration**: Designed for calibrating single-basin models using one USGS streamgage. Multi-gage or multi-objective calibration is not supported.

4. **Custom ngen-cal Branch**: Uses a modified version of ngen-cal from the `ngiab_cal` branch, which differs from the main branch. Features from newer releases of ngen-cal may not be available.

5. **Compatibility Concerns**: The customized ngen-cal implementation may not be compatible with future changes to either ngen-cal or the NGIAB framework.


## Development

Contributions welcome! Comprehensive development instructions coming soon

## Visualisation
During calibration, plots will be created inside `./calibration/Output/Calibration_Run/ngen_*******_worker/Plot_Iteration/`
After calibration, the tool will copy the validated parameters back into the root config folder which allows you to run ngiab with the [guide script](https://github.com/CIROH-UA/NGIAB-CloudInfra/blob/main/guide.sh) and use the teehr / tethys visualiser.
More streamlined workflow coming soon.


## License

"Software code created by U.S. Government employees is not subject to copyright
in the United States (17 U.S.C. §105). The United States/Department of Commerce
reserve all rights to seek and obtain copyright protection in countries other
than the United States for Software authored in its entirety by the Department
of Commerce. To this end, the Department of Commerce hereby grants to Recipient
a royalty-free, nonexclusive license to use, copy, and create derivative works
of the Software outside of the United States."

## Acknowledgments
 
- [CIROH](https://docs.ciroh.org/) for NextGen In A Box
- [NGEN-CAL](https://github.com/NOAA-OWP/ngen-cal) developers
