
import os
import openai

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

openai.api_key = openai_api_key

common_llm_config = {
    "temperature": 0.8,
    "max_tokens": 1500,
    "n": 1,
    "stop": None,
    "config_list": [
        {"model": "gpt-4o", "api_key": openai_api_key},
    ],
}

immunologist_llm_config = {
    **common_llm_config,
    # "functions": [
    #     {
    #         "name": "generate_ab",
    #         "description": "Generates antibody sequences using the PALM-H3 model.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "antigen_sequence": {"type": "string", "description": "The antigen sequence."},
    #                 "origin_seq": {"type": "string", "description": "The origin heavy chain sequence."},
    #                 "origin_light": {"type": "string", "description": "The origin light chain sequence."},
    #                 "cdrh3_begin": {"type": "integer", "description": "CDR H3 begin index."},
    #                 "cdrh3_end": {"type": "integer", "description": "CDR H3 end index."}
    #             },
    #             "required": ["antigen_sequence", "origin_seq", "origin_light", "cdrh3_begin", "cdrh3_end"]
    #         }
    #     },
    # ],
    # "function_call": "auto",
}

project_lead_critic_llm_config = {
    **common_llm_config,
}

# machine_learning_engineer_llm_config = {
#     **common_llm_config,
# }

machine_learning_engineer_llm_config = {
    **common_llm_config,
    "functions": [
        {
            "name": "generate_ab",
            "description": "Generates antibody sequences…",
            "parameters": {
                "type": "object",
                "properties": {
                    "antigen_sequence": {"type": "string"},
                    "origin_seq": {"type": "string"},
                    "origin_light": {"type": "string"},
                    "cdrh3_begin": {"type": "integer"},
                    "cdrh3_end": {"type": "integer"}
                },
                "required": ["antigen_sequence","origin_seq","origin_light","cdrh3_begin","cdrh3_end"]
            }
        },
        {
            "name": "analyze_ab",
            "description": "Takes a CSV path and runs the antibody analysis script.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_csv_path": {
                        "type": "string",
                        "description": "Path to the CSV file containing the newly generated antibodies."
                    }
                },
                "required": ["result_csv_path"]
            }
        },
    ],
    "function_call": "auto",
}
