# llm_config.py
import os
import openai

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

openai.api_key = openai_api_key

# LLM configurations
llm_config = {
    "functions": [
        {
            "name": "generate_antibody_sequence_palm_h3",
            "description": "Generates antibody sequences using the PALM-H3 model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "antigen_sequence": {
                        "type": "string",
                        "description": "The antigen sequence."
                    },
                    "origin_seq": {
                        "type": "string",
                        "description": "The origin heavy chain sequence."
                    },
                    "origin_light": {
                        "type": "string",
                        "description": "The origin light chain sequence."
                    },
                    "cdrh3_begin": {
                        "type": "integer",
                        "description": "CDR H3 begin index."
                    },
                    "cdrh3_end": {
                        "type": "integer",
                        "description": "CDR H3 end index."
                    }
                },
                "required": ["antigen_sequence", "origin_seq", "origin_light", "cdrh3_begin", "cdrh3_end"]
            }
        },        
        {
            "name": "analyze_antibody_properties",
            "description": "Analyzes the properties of an antibody sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "antibody_sequence": {
                        "type": "string",
                        "description": "The antibody sequence to analyze."
                    }
                },
                "required": ["antibody_sequence"]
            }
        },
        # {
        #     "name": "optimize_antibody",
        #     "description": "Optimizes an antibody sequence for desired properties.",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "antibody_sequence": {
        #                 "type": "string",
        #                 "description": "The antibody sequence to optimize."
        #             },
        #             "optimization_goals": {
        #                 "type": "string",
        #                 "description": "Goals for optimization, such as increased stability or reduced immunogenicity."
        #             }
        #         },
        #         "required": ["antibody_sequence"]
        #     }
        # },
    ],

    "temperature": 0.7,
    "max_tokens": 1500,
    "n": 1,
    "stop": None,
    "config_list": [
        {"model": "gpt-4o-mini", "api_key": openai_api_key},
        # You can add more model configurations if needed
    ],
}
