# ScaNER-01: A Multi-Tiered Natural Language Processing Framework for Acute Cerebrovascular Event Labeling

Welcome to the **ScaNER-01** repository! This project provides the code for **ScaNER-01**, a multi-tiered natural language processing (NLP) framework designed to automate the labeling of acute cerebrovascular events in electronic health record (EHR) data. The framework processes clinical text to identify and categorize cerebrovascular events with accuracy and efficiency, leveraging rule-based named entity recognition (NER) and other custom algorithms.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Imports](#data-imports)
- [Framework Guide](#framework-guide)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)


---
## Overview
ScaNER-01 is a sequential, four-tier framework:
1. **Rule-based NER (Tier 1)**: Extracts individual stroke events.
2. **48-Hour Rule (Tier 2)**: Applies specific temporal rules to label neuroimaging reports.
3. **ICD Data Processing (ICD data processing)**: Works with ICD data, adapting to user-specified codes.
4. **ICD and Neuroimaging Merge (Tier 3)**: Merges ICD data with neuroimaging records.
5. **Data Cleaning (Tier 4)**: Cleans and finalizes the data, producing unique stroke codes.

Each tier processes data incrementally, resulting in a comprehensive set of labeled cerebrovascular events.

## Getting Started

### Prerequisites
Before using ScaNER-01, ensure you have:
- Python 3.12+ installed
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MSHS-CNIC/ScaNER-01.git
   cd ScaNER-01
2. **Set Up the Environment**:
   Download and create the environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate scaNER-01
  > Note: The `environment.yml` file may contain additional packages beyond those required for this framework.

### Data Imports
Important: To use this repository with your data, modify the data import paths in the code files to ensure they point to your own datasets. This is required for accurate processing and results.

## Framework Guide
To run the framework, follow the five Python files sequentially:

- **`1_individual_stroke_labeling_algorithm.py`**: Contains the Tier 1 rule-based NER algorithm. Customize terms, rules, getters, and patterns as needed.

- **`2_48hr_stroke_labeling_algorithm.py`**: Implements the 48-hour rule for neuroimaging report labeling (Tier 2).

- **`3_icd_data_process.py`**: Primarily tailored for internal ICD data (Optional), but users can leverage the `stroke_icd_codes` function for custom datasets.

- **`4_icd_scans_caboodle_merge.py`**: Sets up merging rules for ICD codes and neuroimaging, representing the Tier 3 algorithm.

- **`5_df_cleaning.py`**: Performs final data cleaning, identifies unique stroke codes, and generates final labels and dates (Tier 4).

## File Structure

- **`util/`**: Contains auxiliary functions for internal use. Users do not need to interact with these files.

## Contributing
We welcome contributions! If you’d like to contribute, please submit a pull request and ensure your changes are thoroughly documented and tested.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.











