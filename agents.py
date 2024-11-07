#!/usr/bin/env python
# coding: utf-8

### on hold: RAG and optimization modules

import os
import openai
from autogen import AssistantAgent, UserProxyAgent
from llm_config import llm_config
import agent_functions as func
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import ChatCompletion

# # for detailed logging to understand 
# ChatCompletion.start_logging()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="UserProxyAgent",
    system_message=(
        "You are the UserProxyAgent, representing the user in the system. "
        "You interact with other agents by providing input, approving plans, and giving feedback. "
        "Ensure that the agents' actions align with the user's goals and preferences. "
        "Be proactive in requesting clarifications or additional information when needed."
        "Provide detailed output of every single process."
    ),
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False},  # Disable Docker for now
    llm_config=llm_config,
)

# Planner Agent
planner = AssistantAgent(
    name="Planner",
    system_message=(
        "You are the Planner Agent. Your role is to develop a detailed plan to achieve the user's objectives in antibody discovery. "
        "Begin by understanding the user's input and requirements. "
        "Outline the steps needed, including identifying the necessary input parameters (antigen_sequence, origin_seq, origin_light, cdrh3_begin, cdrh3_end), "
        "specifying that the Assistant should call the 'generate_antibody_sequence_palm_h3' function, "
        "and then call 'analyze_antibody_properties' with the generated sequences. "
        "Do not include optimization steps in the plan. " # for now
        "Ensure the plan is clear, actionable, and aligned with the goals."
    ),
    llm_config=llm_config,
)

# Critic Agent
critic = AssistantAgent(
    name="Critic",
    system_message=(
        "You are the Critic Agent. Your role is to critically analyze the plans developed by the Planner. "
        "Check for completeness, correctness, and feasibility, focusing on the functions and parameters used. "
        "Provide constructive feedback to improve the plan, and ensure that it meets the objectives. "
        "When the plan is satisfactory, indicate approval by stating 'TERMINATE'."
    ),
    llm_config=llm_config,
)

# Assistant Agent with function maps
# can consider adding more print statements like 'calling analyze_antibody_properties func' later on
assistant = AssistantAgent(
    name="Assistant",
    system_message=(
        "You are the Assistant Agent with access to custom functions for antibody discovery. "
        "Your role is to execute functions as suggested by the Planner, prepare the required input parameters, and process the results. "
        "When calling 'generate_antibody_sequence_palm_h3' and 'analyze_antibody_properties', ensure you provide all necessary parameters. "
        "Do not execute optimization functions at this time. " # for now
        "Collaborate effectively with other agents, share insights, and contribute to achieving the overall objectives."
    ),
    function_map={
        "generate_antibody_sequence_palm_h3": func.generate_antibody_sequence_palm_h3,
        "analyze_antibody_properties": func.analyze_antibody_properties,
        # "optimize_antibody": func.optimize_antibody,
    },
    llm_config=llm_config,
)

# Executor Agent (if needed)
executor = UserProxyAgent(
    name="Executor",
    system_message=(
        "You are the Executor Agent. Your role is to execute the code provided by the Coder. "
        "Run the code in a safe and controlled environment, capture any outputs or errors, and report back the results. "
        "Ensure that the execution is secure and does not violate any policies. "
        "Provide clear and detailed feedback on the execution outcome."
    ),
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 12,
        "work_dir": "./code_antibody/",
        "use_docker": False  # Not using Docker for now
    },
    llm_config=llm_config,
)

# Add the main function
def main():
    # Step 1: User provides the objective
    user_input = input("Welcome to the Antibody Design Assistant.\nPlease describe your objective:\n")

    # Step 2: Initiate chat between user_proxy and planner
    user_proxy.initiate_chat(
        planner,
        message=user_input,
        conversation_id="antibody_design_conversation"
    )

if __name__ == "__main__":
    main()