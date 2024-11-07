from analyze import analyze_antibody_properties

# Sample heavy and light chain sequences
heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISWNSGSFTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYC"
light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSYLNWYQQKPGKAPKLLIYAASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

# Call the function
result = analyze_antibody_properties(heavy_chain, light_chain)

# Print the results
print(f"Predicted structure saved as: {result['output_pdb']}")
print(f"Visualization saved as: {result['visualization_html']}")
print(f"Aggregation Propensity: {result['aggregation_propensity']}")