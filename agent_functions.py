# agent_functions.py
#!/usr/bin/env python
# coding: utf-8

import json
import torch
import os
import openai
import subprocess
import glob
import pandas as pd
import shutil
import uuid
from datetime import datetime

from llm_config import immunologist_llm_config, project_lead_critic_llm_config, machine_learning_engineer_llm_config
from analyze import analyze_heavy_light as analyze_antibody

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_ab(
    antigen_sequence,
    origin_seq,
    origin_light,
    cdrh3_begin,
    cdrh3_end
):
    """
    This function updates seq2seq_generate.json with input parameters, then runs the generation script.
    It expects to produce only the top 5 candidate sequences (after code changes in generate_antibody.py).
    """
    print("Starting generate_ab function...")

    palm_code_dir = os.path.join(BASE_DIR, 'genab', 'PALM', 'Code')
    config_path = os.path.join(palm_code_dir, 'config', 'common', 'seq2seq_generate.json')

    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Update config with provided parameters
    config_data['origin_seq'] = origin_seq
    config_data['origin_light'] = origin_light
    config_data['cdrh3_begin'] = cdrh3_begin
    config_data['cdrh3_end'] = cdrh3_end
    config_data['use_antigen'] = antigen_sequence

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    # Run the generation script
    generate_script = os.path.join(palm_code_dir, 'generate_antibody.py')
    command = ['python', generate_script, '--config', config_path]
    try:
        subprocess.run(command, cwd=palm_code_dir, check=True, capture_output=True, text=True)
        print("Generation script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Generation failed:\n{e.stderr}")
        return None

    result_base_dir = os.path.join(BASE_DIR, 'genab', 'PALM', 'Result_seq2seq_gen', 'datasplit', 'CoV_AbDab-Seq2seq-Evaluate-Common')
    dirs = glob.glob(os.path.join(result_base_dir, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("No result directories found.")
        return None

    latest_dir = max(dirs, key=os.path.getmtime)
    result_file = os.path.join(latest_dir, 'result.csv')
    if not os.path.exists(result_file):
        print(f"No result.csv found in {latest_dir}")
        return None

    return result_file

def analyze_ab(result_csv_path):
    print("Starting analyze_ab function...")

    palm_code_dir = os.path.join(BASE_DIR, 'genab', 'PALM', 'Code')
    eval_config_path = os.path.join(palm_code_dir, 'config', 'common', 'bert_eval_generation.json')

    directory_of_results = os.path.dirname(result_csv_path)
    with open(eval_config_path, 'r') as f:
        eval_config = json.load(f)

    eval_config['data_loader']['args']['data_dir'] = directory_of_results

    with open(eval_config_path, 'w') as f:
        json.dump(eval_config, f, indent=4)

    eval_script = os.path.join(palm_code_dir, 'eval_generate_seq.py')
    # try:
    #     subprocess.run(['python', eval_script, '--config', eval_config_path], cwd=palm_code_dir, check=True, capture_output=True, text=True)
    #     print("Analysis script executed successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Analysis failed:\n{e.stderr}")
    #     return "Analysis failed."

    # return "Analysis completed successfully."
    try:
        subprocess.run(
            ['python', eval_script, '--config', eval_config_path],
            cwd=palm_code_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("Analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed:\n{e.stderr}")
        return "Analysis failed."

    return "Analysis completed successfully."

# def retrieve_top_5_candidates(result_csv_path):
#     """
#     If needed, parse the result.csv to select top 5 candidates.
#     If we had a scoring metric, we would sort by it here and pick the best 5.
#     """
#     df = pd.read_csv(result_csv_path)
#     # If there's a score, we would do something like: df.sort_values(by='Score', ascending=False).head(5)
#     # For now, we just return the entire df since it's already top 5.
#     return df
