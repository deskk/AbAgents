#!/usr/bin/env python
# coding: utf-8
'''
agent_function.py and agents.py have to be in the same directory
'''

import os
import openai
from autogen import AssistantAgent, UserProxyAgent
from llm_config import llm_config
import agent_functions as func
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the indexed data using llama_index for RAG
DATA_DIR = os.getenv('DATA_DIR', 'data/antibody_antigen_models')
documents = SimpleDirectoryReader(DATA_DIR).load_data()
index = GPTSimpleVectorIndex(documents)
query_engine = index.as_query_engine(similarity_top_k=10)

# Termination message function
termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Define the UserProxyAgent
user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user_proxy. Plan execution needs to be approved by user_proxy.",
    max_consecutive_auto_reply=None,
    code_execution_config=False,
)

# Define the Coder Agent
coder = AssistantAgent(
    name="Coder",
    system_message=(
        "Coder. You write Python code to perform antibody design tasks. "
        "Wrap the code in a code block that specifies the script type. "
        "Do not include multiple code blocks in one response. "
        "Check the execution result returned by the executor. "
        "If there's an error, fix it and output the code again. "
        "Provide the full code instead of partial code or code changes. "
        "Do not install packages."
    ),
    llm_config=llm_config,
)

# Define the Critic Agent
critic = AssistantAgent(
    name="Critic",
    system_message=(
        "Critic. You double-check the plan, especially the functions and function parameters. "
        "Check whether the plan includes all necessary parameters for the suggested function. "
        "Provide feedback. Print TERMINATE when the task is finished successfully."
    ),
    llm_config=llm_config,
)

# Define the Executor Agent
executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. You follow the plan. Execute the code written by the coder and return outcomes.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 12, "work_dir": "./code_antibody/"},
    llm_config=llm_config,
)

# Define the Planner Agent
planner = AssistantAgent(
    name="Planner",
    system_message=(
        "Planner. You develop a plan. Begin by explaining the plan. Revise the plan based on feedback from the critic and user_proxy, until user_proxy approval. "
        "The plan may involve calling custom functions for retrieving knowledge, designing antibodies, and computing and analyzing antibody properties. "
        "Include the function names in the plan and the necessary parameters."
    ),
    llm_config=llm_config,
)

# Define the RetrieveUserProxyAgent (for RAG)
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    system_message="Assistant with extra content retrieval power for antibody domain knowledge. The assistant follows the plan.",
    human_input_mode="NEVER",
    is_termination_msg=termination_msg,
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": f"{DATA_DIR}",
        "chunk_token_size": 3000,
        "model": llm_config['config_list'][0]["model"],
        "chunk_mode": "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
        "get_or_create": True
    },
    llm_config=llm_config,
)

# Define the AntibodyDesignAgent (Assistant Agent with function mapping)
assistant = AssistantAgent(
    name="assistant",
    system_message=(
        "assistant. You have access to all the custom functions. "
        "You focus on executing the functions suggested by the planner or the critic. "
        "You also have the ability to prepare the required input parameters for the functions."
    ),
    llm_config=llm_config,
    function_map={
        "retrieve_antigen_data": func.retrieve_antigen_data,
        # "design_antibody": func.design_antibody,
        "generate_antibody_sequences": func.generate_antibody_sequence,
        "optimize_antibody": func.optimize_antibody,
        "analyze_antibody_properties": func.analyze_antibody_properties,
        # Add more function mappings as needed
    },
)

'''
responsible for orchestrating the workflow between agents
'''
# Define the Coordinator to manage agent interactions
class Coordinator:
    def __init__(self, user_proxy, planner, coder, critic, executor, assistant, ragproxyagent):
        self.user_proxy = user_proxy
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.executor = executor
        self.assistant = assistant
        self.ragproxyagent = ragproxyagent
    
    # Q: does 'chat' between agents work?
    def run(self):
        # Step 1: Get user input
        user_input = self.user_proxy.get_user_input()

        # Step 2: Planner creates a plan
        plan = self.planner.chat(user_input)

        # Step 3: Critic reviews the plan
        critique = self.critic.chat(plan)

        # Step 4: User approves the plan
        approval = self.user_proxy.chat(critique)

        # Step 5: Coder generates code based on the plan
        code = self.coder.chat(approval)

        # Step 6: Executor executes the code
        execution_result = self.executor.chat(code)

        # Step 7: Assistant processes the execution result
        assistant_result = self.assistant.chat(execution_result)

        # Step 8: Return the final result
        print("Antibody Design Output:")
        print(assistant_result)

# Main execution
def main():
    coordinator = Coordinator(
        user_proxy=user_proxy,
        planner=planner,
        coder=coder,
        critic=critic,
        executor=executor,
        assistant=assistant,
        ragproxyagent=ragproxyagent,
    )
    coordinator.run()

if __name__ == "__main__":
    main()
