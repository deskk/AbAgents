'''
in-silico analysis & APIs
'''

import os
import pandas as pd
import requests
import json
import time
import numpy as np
from igfold import IgFoldRunner
from igfold.utils.visualize import show_pdb

# use igfoldrunner to predict the antibody structure from heavy and light chain sequences
def predict_antibody_structure(sequences, output_pdb='predicted_antibody.pdb', do_refine=False, use_openmm=False, do_renum=False, visualize=True):
    '''This function uses the heavy and light chain sequences to predict the antibody structure.'''
    igfold = IgFoldRunner()
    out = igfold.fold(
        output_pdb,
        sequences=sequences,
        do_refine=do_refine,
        use_openmm=use_openmm,
        do_renum=do_renum
    )
    print("Output PDB saved to the following location:", output_pdb)

    if visualize:
        view = show_pdb(output_pdb, len(sequences), bb_sticks=False, sc_sticks=True, color="rainbow")
        visualization_html = os.path.splitext(output_pdb)[0] + '_visualization.html'
        view.write_html(visualization_html)
        print("Visualization saved as:", visualization_html)
    return out

def predict_aggregation_propensity(pdb_file_path):
    '''Predicts the aggregation propensity of a given antibody structure using an external API.'''
    url = 'https://biocomp.chem.uw.edu.pl/A3D2/RESTful/submit/userinput/'
    options = {
        'dynamic': False,
        'distance': 5,
        'name': 'REST_test',
        'hide': True,
        'foldx': True
    }
    
    with open(pdb_file_path, 'rb') as pdb_file:
        files = {
            'inputfile': ("file", pdb_file),
            'json': (None, json.dumps(options), 'application/json')
        }
        response = requests.post(url, files=files)
        
        if response.status_code != 200:
            print(f"Error submitting job: {response.status_code}")
            return None
        
        job_id = response.json().get('jobid')
    
    status_url = f'https://biocomp.chem.uw.edu.pl/A3D2/RESTful/job/{job_id}/status/'
    
    while True:
        response = requests.get(status_url)
        if response.status_code == 200:
            status = response.json().get('status')
            if status == "done":
                break
            elif status == "error":
                print("Job encountered an error.")
                return None
        else:
            print(f"Error checking job status: {response.status_code}")
        time.sleep(5)
    
    result_url = f'https://biocomp.chem.uw.edu.pl/A3D2/RESTful/job/{job_id}/structure/'
    response = requests.get(result_url)
    
    if response.status_code == 200:
        pdb_content = response.text.splitlines()
        a3d_scores = [float(line[60:66]) for line in pdb_content if line.startswith('ATOM')]
        
        if not a3d_scores:
            print("Error: No A3D scores found in the result file.")
            return None
        
        avg_a3d_score = np.mean(a3d_scores)
        return avg_a3d_score
    else:
        print(f"Error retrieving results: {response.status_code}")
        return None

def analyze_antibody_properties(heavy_chain, light_chain):
    # Create 'analysis_output' directory if it doesn't exist
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    output_pdb = os.path.join(output_dir, 'predicted_antibody.pdb')

    # Predict antibody structure
    predict_antibody_structure({'H': heavy_chain, 'L': light_chain}, output_pdb, visualize=True)

    # Calculate aggregation propensity
    agg_propensity = predict_aggregation_propensity(output_pdb)

    # Paths to output files
    visualization_html = os.path.splitext(output_pdb)[0] + '_visualization.html'

    result = {
        'output_pdb': output_pdb,
        'aggregation_propensity': agg_propensity,
        'visualization_html': visualization_html
    }
    return result

# def process_antibody(input_file):

#     '''This is the master function which is called when the this python script "final analyze.py" is executed.
#        It takes in input as a csv file with Antibody heavy and light chain as inputs calls other function to analyze antibody properties.'''

#     # Read input file
#     df = pd.read_csv(input_file)
    
#     # Extract sequences
#     heavy_chain = df['Heavy Chain'].iloc[0]
#     light_chain = df['Light Chain'].iloc[0]
    
#     # Prepare sequences for structure prediction
#     sequences = {
#         "H": heavy_chain,
#         "L": light_chain
#     }
    
#     # Predict antibody structure
#     output_pdb = 'predicted_antibody.pdb'
#     predict_antibody_structure(sequences, output_pdb, visualize=True)
    
#     # Calculate aggregation propensity
#     agg_propensity = predict_aggregation_propensity(output_pdb)
    
#     if agg_propensity is not None:
#         print(f"Average Aggregation Propensity: {agg_propensity}")
#     else:
#         print("Failed to calculate aggregation propensity.")

#     return output_pdb, agg_propensity

# # Main execution
# input_file = 'dummy_files/antibody_sequences.csv'
# output_pdb, agg_propensity = process_antibody(input_file)

# print(f"Predicted structure saved as: {output_pdb}")
# print(f"Visualization saved as: visualization.html")
# print(f"Aggregation Propensity: {agg_propensity}")