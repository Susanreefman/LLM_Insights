# LLM_Insights

The objective of this project is to create a large language model that can generate text
based on the variables measured by sensors such as a soil moisture sensor and meteoro-
logical data.

## Introduction

This study focuses on developing text-based explanations from sensor outputs, specifically
targeting advice generation from variables like soil moisture data. The main objective is
to construct a large language model capable of generating coherent text based on sensor
measurements. Methods involve analyzing recommendations written by users to other
users, deriving textual sentences from numerical data, and training the language model.
The model achieved an accuracy score of 0.94 in generating adequate sentences but faces
challenges with complex or infrequent vocabulary. While the model demonstrates func-
tionality, further refinement is necessary to enhance its linguistic capabilities. Moreover,
exploring different applications or features of large language models may yield improved
results for diverse tasks

## Project Overview
The project is organized the following:

- `LLM.py`: Large language model, included also training and testing implementation
- `LLM_notebook.ipynb`: Large Language model in jupyter notebook format
- `generate_insights.py`: Script to generate textual input sentence for the model
- `sample.csv`: A sample CSV file for testing and demonstration purposes.

## Installation
To use this script, please follow the steps stated below.

**Step 1: Acquiring Files**

Either [clone](https://github.com/Susanreefman/LLM_Insights/blob/main/sample.csv) or generate sentences and corresponding summary manually.

**Step 2: Installing Python**

The script was developed in the language Python (version 3.7.9). Please follow the instructions on how to install Python [here](https://docs.python.org/3/index.html).

### Installing Necessary Packages

In addition, a necessary external packages needs to be installed; torch, tensorflow, scikit-learn and evaluate

```bash
python3 -m pip install [package]
```
Now, you are set to use this program.

## Usage
Use the following steps to run the program. 

### Using the sample file

If you are using the provided sample file, execute the following commands:

To run the model program with the sample file
```bash
$ python3 LLM.py -f sample.csv
```

## Contact
If you have any questions, suggestions, or encounter issues, feel free to reach out:
 [email](mailto:h.s.reefman@st.hanze.nl)
