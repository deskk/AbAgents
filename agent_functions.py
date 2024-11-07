#!/usr/bin/env python
# coding: utf-8

import json
import torch
import os
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import openai
from llm_config import llm_config
import subprocess
import glob
import pandas as pd
import shutil
import uuid

# analyze.py
from analyze import analyze_antibody_properties as analyze_antibody

''' 
Generate antibody sequence using PALM-H3 model.

Requires 5 inputs:
- antigen_sequence
- origin_seq
- origin_light
- cdrh3_begin
- cdrh3_end

The function modifies the generation config file with the inputs,
runs the generation script, and returns the result as a DataFrame.
'''
def generate_antibody_sequence_palm_h3(
    antigen_sequence,
    origin_seq, # original heavy chain
    origin_light,
    cdrh3_begin,
    cdrh3_end
):
    print("Starting generate_antibody_sequence_palm_h3 function...")
    print("Preparing to run the PALM-H3 generation script in the 'palmh3' conda environment.")

    import subprocess
    import json
    import os
    import glob
    import pandas as pd
    import shutil
    import uuid

    # Paths
    original_config_path = '/home/desmond/abagents/generation_palmh3/PALM/Code/config/common/seq2seq_generate.json'
    generation_script_path = '/home/desmond/abagents/generation_palmh3/PALM/Code/generate_antibody.py'
    palm_env_name = 'palmh3'  # conda env for palmh3

    # Create a unique temporary directory to avoid conflicts
    temp_run_id = str(uuid.uuid4())
    temp_dir = os.path.join('/tmp', f'palmh3_run_{temp_run_id}')
    os.makedirs(temp_dir, exist_ok=True)
    temp_config_path = os.path.join(temp_dir, 'seq2seq_generate.json')

    # Copy the config file to the temporary directory
    shutil.copyfile(original_config_path, temp_config_path)

    # Read and modify the config file
    with open(temp_config_path, 'r') as file:
        config_data = json.load(file)

    # Update the required fields
    config_data['origin_seq'] = origin_seq
    config_data['origin_light'] = origin_light
    config_data['cdrh3_begin'] = cdrh3_begin
    config_data['cdrh3_end'] = cdrh3_end
    config_data['use_antigen'] = antigen_sequence

    # Update the save directory to the temporary directory
    config_data['trainer']['save_dir'] = temp_dir

    # Write the modified config back to the temporary file
    with open(temp_config_path, 'w') as file:
        json.dump(config_data, file, indent=4)

    # Run the generation script using subprocess and activate the palmh3 environment
    command = [
        'conda', 'run', '-n', palm_env_name,
        'python', generation_script_path,
        '--config', temp_config_path
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print("Generation script executed successfully.")

        # Locate the result file
        result_files = glob.glob(os.path.join(temp_dir, '**', 'result.csv'), recursive=True)
        if result_files:
            latest_result_file = max(result_files, key=os.path.getctime)
            df = pd.read_csv(latest_result_file)
            # Return the DataFrame
            return df
        else:
            print("No result files found.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e.stderr}")
        return None
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


# # Load the indexed data using llama_index for RAG (if needed)
# DATA_DIR = os.getenv('DATA_DIR', 'data/antibody_antigen_models')
# documents = SimpleDirectoryReader(DATA_DIR).load_data()
# index = VectorStoreIndex(documents)
# query_engine = index.as_query_engine()

# # Function to retrieve antigen data using llama_index
# def retrieve_antigen_data(antigen_name):
#     print(f"Retrieving data for antigen: {antigen_name}")
#     response = query_engine.query(f"Information about antigen {antigen_name}")
#     antigen_info = response.response
#     return antigen_info

# add error handling later
def analyze_antibody_properties(heavy_chain, light_chain):
    '''
    Analyzes the properties of an antibody's heavy and light chain sequences.
    '''
    # Call the analysis function from analyze.py
    result = analyze_antibody(heavy_chain, light_chain)
    
    # Process the result and format the analysis report
    if result['aggregation_propensity'] is not None:
        analysis_report = (
            f"Predicted structure saved as: {result['output_pdb']}\n"
            f"Visualization saved as: {result['visualization_html']}\n"
            f"Average Aggregation Propensity: {result['aggregation_propensity']}"
        )
    else:
        analysis_report = "Failed to calculate aggregation propensity."
    
    return analysis_report

# Function to optimize an antibody sequence (placeholder for future implementation)
def optimize_antibody(antibody_sequence, optimization_goals=''):
    # Placeholder implementation
    return "Optimized antibody sequence (functionality to be implemented)"

'''
next steps including:
- temporary directory cleanup
- rag
- optimization
'''
