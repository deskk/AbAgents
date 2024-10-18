#!/usr/bin/env python
# coding: utf-8

"""
abgen.py

Script for generating antibody sequences using the PALM-H3 model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load the PALM-H3 model and tokenizer
def load_palm_h3_model(model_dir):
    """
    Loads the PALM-H3 model and tokenizer from the specified directory.

    Args:
        model_dir (str): The directory where the PALM-H3 model is stored.

    Returns:
        model (AutoModelForSeq2SeqLM): The loaded PALM-H3 model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)

    return model, tokenizer

# Function to generate antibody sequences using PALM-H3
def generate_antibody_sequence_palm_h3(
    antigen_sequence,
    heavy_chain_sequence,
    light_chain_sequence,
    cdrh3_begin,
    cdrh3_end,
    requirements='',
    model=None,
    tokenizer=None
):
    """
    Generates an antibody sequence targeting the given antigen using PALM-H3.

    Args:
        antigen_sequence (str): The amino acid sequence of the antigen.
        heavy_chain_sequence (str): The amino acid sequence of the heavy chain.
        light_chain_sequence (str): The amino acid sequence of the light chain.
        cdrh3_begin (int): The starting index of the CDRH3 region in the heavy chain.
        cdrh3_end (int): The ending index of the CDRH3 region in the heavy chain.
        requirements (str): Additional requirements or constraints for the antibody design.
        model (AutoModelForSeq2SeqLM): The PALM-H3 model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.

    Returns:
        generated_sequence (str): The generated antibody heavy chain sequence with the new CDRH3 region.
    """
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided.")

    device = next(model.parameters()).device

    # Prepare the input text as per PALM-H3's requirements
    # Note: Adjust the input format if necessary based on PALM-H3's expected input
    input_text = f"{antigen_sequence} [SEP] {heavy_chain_sequence} [SEP] {light_chain_sequence} [SEP] {cdrh3_begin} [SEP] {cdrh3_end} [SEP] {requirements}"

    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate the antibody sequence
    outputs = model.generate(
        input_ids=input_ids,
        max_length=512,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the generated sequence
    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_sequence

# Main execution
if __name__ == "__main__":
    # Set the model directory (update this path to your model location)
    model_dir = '/path/to/your/palm_h3_model'  # Replace with the actual path

    # Load the PALM-H3 model and tokenizer
    try:
        model, tokenizer = load_palm_h3_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Example input data (replace with your actual data)
    antigen_sequence = "ANTIGEN_SEQUENCE_HERE"  # Replace with the antigen amino acid sequence
    heavy_chain_sequence = "HEAVY_CHAIN_SEQUENCE_HERE"  # Replace with the heavy chain amino acid sequence
    light_chain_sequence = "LIGHT_CHAIN_SEQUENCE_HERE"  # Replace with the light chain amino acid sequence
    cdrh3_begin = 99  # Replace with the actual CDRH3 begin index
    cdrh3_end = 111  # Replace with the actual CDRH3 end index
    requirements = "High affinity and low immunogenicity"  # Replace with any design requirements

    # Generate the antibody sequence
    try:
        generated_sequence = generate_antibody_sequence_palm_h3(
            antigen_sequence=antigen_sequence,
            heavy_chain_sequence=heavy_chain_sequence,
            light_chain_sequence=light_chain_sequence,
            cdrh3_begin=cdrh3_begin,
            cdrh3_end=cdrh3_end,
            requirements=requirements,
            model=model,
            tokenizer=tokenizer
        )

        print("Generated Antibody Sequence:")
        print(generated_sequence)

    except Exception as e:
        print(f"Error generating antibody sequence: {e}")
