#!/usr/bin/env python
# coding: utf-8

import json
import torch
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import openai
from llm_config import llm_config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

'''
generative model for antibody design conditioned on antigen
'''
# Load the AA model and tokenizer
def load_palm_h3_model(model_dir, antibody_tokenizer_dir, antigen_tokenizer_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizers
    antibody_tokenizer = AutoTokenizer.from_pretrained(antibody_tokenizer_dir)
    antigen_tokenizer = AutoTokenizer.from_pretrained(antigen_tokenizer_dir)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)

    return model, antibody_tokenizer, antigen_tokenizer

def generate_antibody_sequence_palm_h3(antigen_sequence, heavy_chain_sequence, light_chain_sequence, cdrh3_begin, cdrh3_end):
    # Load the model and tokenizers if not already loaded
    if not hasattr(generate_antibody_sequence_palm_h3, 'model'):
        base_dir = 'palm-model/Model_Zenodo'
        model_dir = os.path.join(base_dir, 'PALM_seq2seq')
        antibody_tokenizer_dir = os.path.join(base_dir, 'Heavy_roformer')
        antigen_tokenizer_dir = os.path.join(base_dir, 'antigenmodel')
        generate_antibody_sequence_palm_h3.model, generate_antibody_sequence_palm_h3.antibody_tokenizer, generate_antibody_sequence_palm_h3.antigen_tokenizer = load_palm_h3_model(
            model_dir, antibody_tokenizer_dir, antigen_tokenizer_dir
        )

    model = generate_antibody_sequence_palm_h3.model
    antibody_tokenizer = generate_antibody_sequence_palm_h3.antibody_tokenizer
    antigen_tokenizer = generate_antibody_sequence_palm_h3.antigen_tokenizer

    device = next(model.parameters()).device

    # Tokenize sequences
    antigen_ids = antigen_tokenizer.encode(antigen_sequence, add_special_tokens=False)
    heavy_ids = antibody_tokenizer.encode(heavy_chain_sequence, add_special_tokens=False)
    light_ids = antibody_tokenizer.encode(light_chain_sequence, add_special_tokens=False)

    # Convert CDR H3 indices to tensors
    cdrh3_begin_tensor = torch.tensor([cdrh3_begin], device=device)
    cdrh3_end_tensor = torch.tensor([cdrh3_end], device=device)

    # Prepare input IDs and attention masks
    inputs = {
        'input_ids': torch.tensor([heavy_ids], device=device),
        'attention_mask': torch.ones((1, len(heavy_ids)), device=device),
        'decoder_input_ids': torch.tensor([light_ids], device=device),
        'antigen_input_ids': torch.tensor([antigen_ids], device=device),
        'cdr3_start': cdrh3_begin_tensor,
        'cdr3_end': cdrh3_end_tensor,
    }

    # Generate the antibody sequence
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
    )

    # Decode output
    generated_sequence = antibody_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence


# Load the indexed data using llama_index for RAG
DATA_DIR =os.getenv('DATA_DIR', 'data/antibody_antigen_models')
documents = SimpleDirectoryReader(DATA_DIR).load_data()
index = GPTSimpleVectorIndex(documents)
query_engine = index.as_query_engine(similarity_top_k=10)

# Function to retrieve antigen data using llama_index
def retrieve_antigen_data(antigen_name):
    print(f"Retrieving data for antigen: {antigen_name}")
    response = query_engine.query(f"Information about antigen {antigen_name}")
    antigen_info = response.response
    return antigen_info

'''
to improve specific properties of an existing antibody seq

(tentative) optimization scheme:
- * Binding affinity: Enhancing the binding strength between the antibody and its antigen

- Solubility: Improving the solubility of the antibody
- Aggregation propensity: Reducing the tendency to form aggregates
- Humanization: Minimizing non-human sequences

?? - Stability: Improving the structural integrity under physiological conditions
?? - Specificity: Reducing off-target interactions
?? - Immunogenicity: Minimizing the potential to elicit an unwanted immune response
?? - Developability: Enhancing manufacturability and pharmacokinetic properties
'''
# Function to optimize an antibody sequence
def optimize_antibody(antibody_sequence, optimization_goals=''):
    prompt = (
        f"Optimize the following antibody sequence:\n{antibody_sequence}\n\n"
        f"Optimization Goals:\n{optimization_goals}\n"
        f"Provide the optimized antibody sequence and explain the modifications."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in antibody optimization."},
            {"role": "user", "content": prompt}
        ],
        **llm_config['openai_params']
    )
    optimized_antibody = response['choices'][0]['message']['content']
    return optimized_antibody

'''
evaluates antibody seq to predict its biophysical and biochemical properties:
# - binding affinity, solubiility, aggregation propensity, humanization

?? - Efficacy: Potential effectiveness in neutralizing the target antigen.
?? - Stability: Thermal and chemical stability predictions.
?? - Immunogenicity: Likelihood of triggering an immune response.
?? - Aggregation Propensity: Tendency to form aggregates, which is undesirable.
'''
# Function to analyze antibody properties
def analyze_antibody_properties(antibody_sequence):
    prompt = (
        f"Analyze the following antibody sequence:\n{antibody_sequence}\n\n"
        f"Provide a detailed analysis of its properties, including binding affinity, solubility, aggregation propensity, and humanization."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in antibody analysis."},
            {"role": "user", "content": prompt}
        ],
        **llm_config['openai_params']
    )
    analysis = response['choices'][0]['message']['content']
    return analysis
