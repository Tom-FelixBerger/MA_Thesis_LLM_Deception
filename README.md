# Robust and Transparent Deception Mitigation in LLMs by Aligning Representations of Own and Others' Beliefs
**Author:** Tom-Felix Berger  

This repository contains the code and data for all four experiments conducted as part of the author's master's thesis.

## Overview

This project investigates deception mitigation in large language models (LLMs) by aligning internal representations of self- and other-beliefs.  
The repository includes:

- Experiment code (4 experiments)
- Supporting data and utilities
- Scripts to reproduce all results

## Usage

Follow the steps below to reproduce the experiments.


### **1. Install Dependencies**

Install **Python 3.11**, create and activate a virtual environment, then install the required packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.\.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```


### **2. Authenticate with Hugging Face**

You need access to the following models:

- https://huggingface.co/google/gemma-2-9b-it
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Steps:

1. Request model access on Hugging Face.  
2. Create a Hugging Face authentication token.  
3. Duplicate `credentials.txt.stub`, rename it (e.g., `credentials.txt`), and paste your token where indicated.


### **3. Run All Experiments**

From the `src` directory, run:

```bash
python -m all_experiments_driver
```


### **4. Run Experiments Individually (Optional)**

Example for Experiment 1:

```bash
python -m Experiment1.forced_choice_task
```
