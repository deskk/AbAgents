#!/usr/bin/env python
# coding: utf-8

import os

import openai

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

openai.api_key = openai_api_key

# LLM configurations
llm_config = {
    "functions": [
        # {
        #     "name": "design_antibody",
        #     "description": "Designs an antibody sequence targeting a specific antigen.",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "antigen_name": {
        #                 "type": "string",
        #                 "description": "Name of the antigen."
        #             },
        #             "requirements": {
        #                 "type": "string",
        #                 "description": "Specific requirements for the antibody, such as high affinity or low immunogenicity."
        #             }
        #         },
        #         "required": ["antigen_name"]
        #     }
        # },
        {
            "name": "optimize_antibody",
            "description": "Optimizes an antibody sequence for desired properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "antibody_sequence": {
                        "type": "string",
                        "description": "The antibody sequence to optimize."
                    },
                    "optimization_goals": {
                        "type": "string",
                        "description": "Goals for optimization, such as increased stability or reduced immunogenicity."
                    }
                },
                "required": ["antibody_sequence"]
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
        # Add more function definitions as needed
    ],
    "openai_params": {
        "temperature": 0.7,
        "max_tokens": 1500, # be mindful of this, as it can be expensive
        "n": 1,
        "stop": None,
    },
    "config_list": [
        {"model": "gpt-3.5-turbo", "api_key": openai_api_key},
        # You can add more model configurations if needed
    ],
}
