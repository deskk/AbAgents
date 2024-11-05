# #!/usr/bin/env python
# # coding: utf-8

# """
# abgen.py

# Script for generating antibody sequences using the PALM-H3 model.
# """

# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Function to load the PALM-H3 model and tokenizers
# def load_palm_h3_model(model_dir, heavy_tokenizer_dir, light_tokenizer_dir, antigen_tokenizer_dir):
#     """
#     Loads the PALM-H3 model and tokenizers from the specified directories.

#     Args:
#         model_dir (str): The directory where the PALM-H3 model is stored.
#         heavy_tokenizer_dir (str): The directory where the heavy chain tokenizer is stored (Heavy_roformer).
#         light_tokenizer_dir (str): The directory where the light chain tokenizer is stored (Light_roformer).
#         antigen_tokenizer_dir (str): The directory where the antigen tokenizer is stored (antigenmodel).

#     Returns:
#         model (AutoModelForSeq2SeqLM): The loaded PALM-H3 model.
#         heavy_tokenizer (AutoTokenizer): The tokenizer for heavy chain sequences.
#         light_tokenizer (AutoTokenizer): The tokenizer for light chain sequences.
#         antigen_tokenizer (AutoTokenizer): The tokenizer for antigen sequences.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Load the heavy chain tokenizer
#     heavy_tokenizer = AutoTokenizer.from_pretrained(heavy_tokenizer_dir, local_files_only=True)
#     # Load the light chain tokenizer
#     light_tokenizer = AutoTokenizer.from_pretrained(light_tokenizer_dir, local_files_only=True)
#     # Load the antigen tokenizer
#     antigen_tokenizer = AutoTokenizer.from_pretrained(antigen_tokenizer_dir, local_files_only=True)

#     # Load the PALM-H3 model
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
#     model.to(device)

#     return model, heavy_tokenizer, light_tokenizer, antigen_tokenizer

# # Function to generate antibody sequences using PALM-H3
# def generate_antibody_sequence_palm_h3(
#     antigen_sequence,
#     heavy_chain_sequence,
#     light_chain_sequence,
#     cdrh3_begin,
#     cdrh3_end,
#     model=None,
#     heavy_tokenizer=None,
#     light_tokenizer=None,
#     antigen_tokenizer=None
# ):
#     """
#     Generates an antibody sequence targeting the given antigen using PALM-H3.

#     Args:
#         antigen_sequence (str): The amino acid sequence of the antigen.
#         heavy_chain_sequence (str): The amino acid sequence of the heavy chain.
#         light_chain_sequence (str): The amino acid sequence of the light chain.
#         cdrh3_begin (int): The starting index of the CDR H3 region in the heavy chain.
#         cdrh3_end (int): The ending index of the CDR H3 region in the heavy chain.
#         model (AutoModelForSeq2SeqLM): The PALM-H3 model.
#         heavy_tokenizer (AutoTokenizer): The tokenizer for heavy chain sequences.
#         light_tokenizer (AutoTokenizer): The tokenizer for light chain sequences.
#         antigen_tokenizer (AutoTokenizer): The tokenizer for antigen sequences.

#     Returns:
#         generated_sequence (str): The generated antibody heavy chain sequence with the new CDR H3 region.
#     """
#     if model is None or heavy_tokenizer is None or light_tokenizer is None or antigen_tokenizer is None:
#         raise ValueError("Model and tokenizers must be provided.")

#     device = next(model.parameters()).device

#     # Tokenize sequences
#     antigen_ids = antigen_tokenizer.encode(antigen_sequence, add_special_tokens=False)
#     heavy_ids = heavy_tokenizer.encode(heavy_chain_sequence, add_special_tokens=False)
#     light_ids = light_tokenizer.encode(light_chain_sequence, add_special_tokens=False)

#     # Convert CDR H3 indices to tensors
#     cdrh3_begin_tensor = torch.tensor([cdrh3_begin], device=device, dtype=torch.long)
#     cdrh3_end_tensor = torch.tensor([cdrh3_end], device=device, dtype=torch.long)

#     # Prepare input IDs and attention masks
#     inputs = {
#         'input_ids': torch.tensor([heavy_ids], device=device),
#         'attention_mask': torch.ones((1, len(heavy_ids)), device=device),
#         'decoder_input_ids': torch.tensor([light_ids], device=device),
#         'decoder_attention_mask': torch.ones((1, len(light_ids)), device=device),
#         'antigen_input_ids': torch.tensor([antigen_ids], device=device),
#         'antigen_attention_mask': torch.ones((1, len(antigen_ids)), device=device),
#         'cdr3_start': cdrh3_begin_tensor,
#         'cdr3_end': cdrh3_end_tensor,
#     }

#     # Generate the antibody sequence
#     outputs = model.generate(
#         **inputs,
#         max_length=512,
#         num_return_sequences=1,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=1.0,
#     )

#     # Decode the generated sequence
#     generated_sequence = heavy_tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return generated_sequence

# # Main execution
# if __name__ == "__main__":
#     import os

#     # Update this path to the absolute path of your Model_Zenodo directory
#     base_dir = os.path.abspath('palm-model/Model_Zenodo')
#     model_dir = os.path.join(base_dir, 'PALM_seq2seq')
#     heavy_tokenizer_dir = os.path.join(base_dir, 'Heavy_roformer')
#     light_tokenizer_dir = os.path.join(base_dir, 'Light_roformer')
#     antigen_tokenizer_dir = os.path.join(base_dir, 'antigenmodel')

#     # Load the PALM-H3 model and tokenizers
#     try:
#         model, heavy_tokenizer, light_tokenizer, antigen_tokenizer = load_palm_h3_model(
#             model_dir, heavy_tokenizer_dir, light_tokenizer_dir, antigen_tokenizer_dir
#         )
#     except Exception as e:
#         print(f"Error loading model and tokenizers: {e}")
#         exit(1)

#     # Example input data (replace with your actual data)
#     antigen_sequence = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCV"
#     heavy_chain_sequence = (
#         "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
#         "SAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR"
#     )
#     light_chain_sequence = (
#         "DIQMTQSPSSLSASVGDRVTITCRASSSVNYLNWYQQKPGKAPKLLIY"
#     )
#     cdrh3_begin = 95  # Adjust based on your heavy chain sequence
#     cdrh3_end = 102   # Adjust based on your heavy chain sequence

#     # Generate the antibody sequence
#     try:
#         generated_sequence = generate_antibody_sequence_palm_h3(
#             antigen_sequence=antigen_sequence,
#             heavy_chain_sequence=heavy_chain_sequence,
#             light_chain_sequence=light_chain_sequence,
#             cdrh3_begin=cdrh3_begin,
#             cdrh3_end=cdrh3_end,
#             model=model,
#             heavy_tokenizer=heavy_tokenizer,
#             light_tokenizer=light_tokenizer,
#             antigen_tokenizer=antigen_tokenizer
#         )

#         print("Generated Antibody Sequence:")
#         print(generated_sequence)

#     except Exception as e:
#         print(f"Error generating antibody sequence: {e}")


#!/usr/bin/env python
# coding: utf-8

"""
abgen.py

Script for generating antibody sequences using the PALM-H3 model.
"""

import os
import torch
from transformers import AutoTokenizer, EncoderDecoderModel, EsmTokenizer

# Function to load the PALM-H3 model and tokenizers
def load_palm_h3_model(model_dir, heavy_tokenizer_dir, light_tokenizer_dir, antigen_model_name):
    """
    Loads the PALM-H3 model and tokenizers from the specified directories.

    Args:
        model_dir (str): The directory where the PALM-H3 model is stored.
        heavy_tokenizer_dir (str): The directory where the heavy chain tokenizer is stored.
        light_tokenizer_dir (str): The directory where the light chain tokenizer is stored.
        antigen_model_name (str): The name of the antigen model (e.g., 'facebook/esm2_t30_150M_UR50D').

    Returns:
        model (EncoderDecoderModel): The loaded PALM-H3 model.
        heavy_tokenizer (AutoTokenizer): The tokenizer for heavy chain sequences.
        light_tokenizer (AutoTokenizer): The tokenizer for light chain sequences.
        antigen_tokenizer (EsmTokenizer): The tokenizer for antigen sequences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizers
    heavy_tokenizer = AutoTokenizer.from_pretrained(heavy_tokenizer_dir, local_files_only=True)
    light_tokenizer = AutoTokenizer.from_pretrained(light_tokenizer_dir, local_files_only=True)
    antigen_tokenizer = EsmTokenizer.from_pretrained(antigen_model_name, do_lower_case=False)

    # Load the PALM-H3 model
    model = EncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device)

    return model, heavy_tokenizer, light_tokenizer, antigen_tokenizer

# Function to split sequences into tokens
def sequence_split_fn(sequence):
    return list(sequence)

# Function to generate antibody sequences using PALM-H3
def generate_antibody_sequence_palm_h3(
    antigen_sequence,
    origin_seq,
    origin_light,
    cdrh3_begin,
    cdrh3_end,
    model=None,
    heavy_tokenizer=None,
    light_tokenizer=None,
    antigen_tokenizer=None,
    num_return_sequences=1
):
    """
    Generates an antibody sequence targeting the given antigen using PALM-H3.

    Args:
        antigen_sequence (str): The amino acid sequence of the antigen.
        origin_seq (str): The original heavy chain sequence.
        origin_light (str): The original light chain sequence.
        cdrh3_begin (int): The starting index of the CDR H3 region in the heavy chain.
        cdrh3_end (int): The ending index of the CDR H3 region in the heavy chain.
        model (EncoderDecoderModel): The PALM-H3 model.
        heavy_tokenizer (AutoTokenizer): The tokenizer for heavy chain sequences.
        light_tokenizer (AutoTokenizer): The tokenizer for light chain sequences.
        antigen_tokenizer (EsmTokenizer): The tokenizer for antigen sequences.
        num_return_sequences (int): Number of sequences to generate.

    Returns:
        result_list (list): A list of dictionaries containing the generated sequences.
    """
    if model is None or heavy_tokenizer is None or antigen_tokenizer is None:
        raise ValueError("Model and tokenizers must be provided.")

    device = next(model.parameters()).device

    # Tokenize the antigen sequence
    antigen_tokenized = antigen_tokenizer(
        antigen_sequence,
        padding="max_length",
        max_length=300,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = antigen_tokenized.input_ids.to(device)
    attention_mask = antigen_tokenized.attention_mask.to(device)

    # Generate sequences
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=18,
        min_length=14,
        num_beams=10,
        do_sample=False,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
    )

    # Decode the generated sequences
    output_str = heavy_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_cdrh3_list = [s.replace(" ", "") for s in output_str if s.strip() != '']

    result_list = []
    for cdrh3 in generated_cdrh3_list:
        # Replace the CDR H3 region in the original heavy chain sequence
        heavy_chain_seq = origin_seq[:cdrh3_begin] + cdrh3 + origin_seq[cdrh3_end:]
        result = {
            'Antigen': antigen_sequence,
            'Generated_CDR_H3': cdrh3,
            'Heavy_Chain': heavy_chain_seq,
            'Light_Chain': origin_light
        }
        result_list.append(result)

    return result_list

# Main execution
if __name__ == "__main__":
    base_dir = 'palm-model/Model_Zenodo'  # Replace with the actual base path
    model_dir = os.path.join(base_dir, 'PALM_seq2seq')
    heavy_tokenizer_dir = os.path.join(base_dir, 'Heavy_roformer')
    light_tokenizer_dir = os.path.join(base_dir, 'Light_roformer')
    antigen_model_name = 'facebook/esm2_t30_150M_UR50D'  # Based on the antigen model's config

    # Load the PALM-H3 model and tokenizers
    try:
        model, heavy_tokenizer, light_tokenizer, antigen_tokenizer = load_palm_h3_model(
            model_dir, heavy_tokenizer_dir, light_tokenizer_dir, antigen_model_name
        )
    except Exception as e:
        print(f"Error loading model and tokenizers: {e}")
        exit(1)

    # Example input data (replace with your actual data)
    antigen_sequence = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYS"
    origin_seq = "QVQLVQSGAEVKKPGSSVNVSCKASGGTLNIYTFSWVRQAPGQGLEWVGTIVPLVGKANYPHKFQGRVTITADKSTSTVNMELSSLRSEDTAVYYCASEVLDNLRDGYNFWGQGTLVTVSS"
    origin_light = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSYLNWYQQKPGKAPKLLIYAVSSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQTYNTLTFGGGTKVEIK"
    cdrh3_begin = 97
    cdrh3_end = 110  # Note: In Python slicing, end index is exclusive

    # Generate the antibody sequences
    try:
        result_list = generate_antibody_sequence_palm_h3(
            antigen_sequence=antigen_sequence,
            origin_seq=origin_seq,
            origin_light=origin_light,
            cdrh3_begin=cdrh3_begin,
            cdrh3_end=cdrh3_end,
            model=model,
            heavy_tokenizer=heavy_tokenizer,
            light_tokenizer=light_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
            num_return_sequences=5  # Generate 5 sequences
        )

        # Print the generated sequences
        for result in result_list:
            print("Antigen:", result['Antigen'])
            print("Generated CDR H3:", result['Generated_CDR_H3'])
            print("Heavy Chain Sequence:", result['Heavy_Chain'])
            print("Light Chain Sequence:", result['Light_Chain'])
            print("-" * 50)

    except Exception as e:
        print(f"Error generating antibody sequence: {e}")
