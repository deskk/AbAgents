from analyze import analyze_antibody_properties

heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISWNSGSFTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYC"
light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSYLNWYQQKPGKAPKLLIYAASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
result = analyze_antibody_properties(heavy_chain, light_chain)

print(f"Predicted structure saved as: {result['output_pdb']}")
print(f"Visualization saved as: {result['visualization_html']}")
print(f"Aggregation Propensity: {result['aggregation_propensity']}")