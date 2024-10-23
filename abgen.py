#!/usr/bin/env python
# coding: utf-8

"""
abgen.py

Script for generating antibody sequences using the PALM-H3 model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load the PALM-H3 model and tokenizers
def load_palm_h3_model(model_dir, antibody_tokenizer_dir, antigen_tokenizer_dir):
    """
    Loads the PALM-H3 model and tokenizers from the specified directories.

    Args:
        model_dir (str): The directory where the PALM-H3 model is stored.
        antibody_tokenizer_dir (str): The directory where the antibody tokenizer is stored (Heavy_roformer).
        antigen_tokenizer_dir (str): The directory where the antigen tokenizer is stored (antigenmodel).

    Returns:
        model (AutoModelForSeq2SeqLM): The loaded PALM-H3 model.
        antibody_tokenizer (AutoTokenizer): The tokenizer for antibody sequences.
        antigen_tokenizer (AutoTokenizer): The tokenizer for antigen sequences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the antibody tokenizer
    antibody_tokenizer = AutoTokenizer.from_pretrained(antibody_tokenizer_dir)
    # Load the antigen tokenizer
    antigen_tokenizer = AutoTokenizer.from_pretrained(antigen_tokenizer_dir)

    # Load the PALM-H3 model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)

    return model, antibody_tokenizer, antigen_tokenizer

# Function to generate antibody sequences using PALM-H3
def generate_antibody_sequence_palm_h3(
    antigen_sequence,
    heavy_chain_sequence,
    light_chain_sequence,
    cdrh3_begin,
    cdrh3_end,
    model=None,
    antibody_tokenizer=None,
    antigen_tokenizer=None
):
    """
    Generates an antibody sequence targeting the given antigen using PALM-H3.

    Args:
        antigen_sequence (str): The amino acid sequence of the antigen.
        heavy_chain_sequence (str): The amino acid sequence of the heavy chain.
        light_chain_sequence (str): The amino acid sequence of the light chain.
        cdrh3_begin (int): The starting index of the CDR H3 region in the heavy chain.
        cdrh3_end (int): The ending index of the CDR H3 region in the heavy chain.
        model (AutoModelForSeq2SeqLM): The PALM-H3 model.
        antibody_tokenizer (AutoTokenizer): The tokenizer for antibody sequences.
        antigen_tokenizer (AutoTokenizer): The tokenizer for antigen sequences.

    Returns:
        generated_sequence (str): The generated antibody heavy chain sequence with the new CDR H3 region.
    """
    if model is None or antibody_tokenizer is None or antigen_tokenizer is None:
        raise ValueError("Model and tokenizers must be provided.")

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

    # Decode the generated sequence
    generated_sequence = antibody_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence

# Main execution
if __name__ == "__main__":
    base_dir = 'palm-model/Model_Zenodo'
    model_dir = os.path.join(base_dir, 'PALM_seq2seq')
    antibody_tokenizer_dir = os.path.join(base_dir, 'Heavy_roformer')
    antigen_tokenizer_dir = os.path.join(base_dir, 'antigenmodel')

    # Load the PALM-H3 model and tokenizers
    try:
        model, antibody_tokenizer, antigen_tokenizer = load_palm_h3_model(
            model_dir, antibody_tokenizer_dir, antigen_tokenizer_dir
        )
    except Exception as e:
        print(f"Error loading model and tokenizers: {e}")
        exit(1)

    # Example input data (replace with your actual data)
    antigen_sequence = "ANTIGEN_SEQUENCE_HERE"  # Replace with the antigen amino acid sequence
    heavy_chain_sequence = "HEAVY_CHAIN_SEQUENCE_HERE"  # Replace with the heavy chain amino acid sequence
    light_chain_sequence = "LIGHT_CHAIN_SEQUENCE_HERE"  # Replace with the light chain amino acid sequence
    cdrh3_begin = 99  # Replace with the actual CDR H3 begin index
    cdrh3_end = 111  # Replace with the actual CDR H3 end index

    # Generate the antibody sequence
    try:
        generated_sequence = generate_antibody_sequence_palm_h3(
            antigen_sequence=antigen_sequence,
            heavy_chain_sequence=heavy_chain_sequence,
            light_chain_sequence=light_chain_sequence,
            cdrh3_begin=cdrh3_begin,
            cdrh3_end=cdrh3_end,
            model=model,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer
        )

        print("Generated Antibody Sequence:")
        print(generated_sequence)

    except Exception as e:
        print(f"Error generating antibody sequence: {e}")
