#!/usr/bin/env python
# coding: utf-8

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
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# UserProxyAgent
user_proxy = UserProxyAgent(
    name="UserProxyAgent",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message=(
        "You are the UserProxyAgent, representing the user in the system. "
        "You interact with other agents by providing input, approving plans, and giving feedback. "
        "Ensure that the agents' actions align with the user's goals and preferences. "
        "Be proactive in requesting clarifications or additional information when needed."
    ),
    max_consecutive_auto_reply=None,
    code_execution_config=False,
)

'''
Emphasized understanding requirements, writing efficient code, and readiness to debug
'''
# Coder Agent
coder = AssistantAgent(
    name="Coder",
    system_message=(
        "You are the Coder Agent. Your role is to write high-quality Python code to perform antibody design tasks as per the plan. "
        "Understand the requirements, write efficient and correct code, and be ready to debug and fix issues based on feedback from the Executor. "
        "Provide code in properly formatted code blocks, and ensure your code is executable and well-documented."
    ),
    llm_config=llm_config,
)

'''
Clarified the role in critically analyzing plans.
Instructed to provide constructive feedback and indicate approval with 'TERMINATE'.
'''
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

# Executor Agent
executor = UserProxyAgent(
    name="Executor",
    system_message=(
        "You are the Executor Agent. Your role is to execute the code provided by the Coder. "
        "Run the code in a safe and controlled environment, capture any outputs or errors, and report back the results. "
        "Ensure that the execution is secure and does not violate any policies. "
        "Provide clear and detailed feedback on the execution outcome."
    ),
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 12, "work_dir": "./code_antibody/"},
    llm_config=llm_config,
)

# Planner Agent
planner = AssistantAgent(
    name="Planner",
    system_message=(
        "You are the Planner Agent. Your role is to develop a detailed plan to achieve the user's objectives in antibody discovery. "
        "Begin by understanding the user's input and requirements. "
        "Outline the steps needed, including calling custom functions and coordinating with other agents. "
        "Be flexible and iterative, revising the plan based on feedback from the Critic and UserProxyAgent. "
        "Ensure the plan is clear, actionable, and aligned with the goals."
    ),
    llm_config=llm_config,
)

# RetrieveUserProxyAgent (for RAG)
ragproxyagent = RetrieveUserProxyAgent(
    name="RetrieveUserProxyAgent",
    system_message=(
        "You are the RetrieveUserProxyAgent with advanced content retrieval capabilities for antibody domain knowledge. "
        "Your role is to provide relevant information from the knowledge base to assist other agents. "
        "Follow the plan and respond to queries with precise and accurate information."
    ),
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

# Assistant Agent (with function mapping)
assistant = AssistantAgent(
    name="Assistant",
    system_message=(
        "You are the Assistant Agent with access to custom functions for antibody discovery. "
        "Your role is to execute functions as suggested by the Planner or Critic, prepare the required input parameters, and process the results. "
        "Collaborate effectively with other agents, share insights, and contribute to achieving the overall objectives."
    ),
    llm_config=llm_config,
    function_map={
        "retrieve_antigen_data": func.retrieve_antigen_data,
        "generate_antibody_sequence_palm_h3": func.generate_antibody_sequence_palm_h3,
        "optimize_antibody": func.optimize_antibody,
        "analyze_antibody_properties": func.analyze_antibody_properties,
    },
)

'''
run method includes loops and conditional checks to allow for iterative refinement of 
the plan and code, error handling, and more interactive communication between agents
'''
# Coordinator class to manage agent interactions
class Coordinator:
    def __init__(self, user_proxy, planner, coder, critic, executor, assistant, ragproxyagent):
        self.user_proxy = user_proxy
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.executor = executor
        self.assistant = assistant
        self.ragproxyagent = ragproxyagent

    def run(self):
        # Step 1: Get user input
        user_input = self.user_proxy.get_user_input()

        # Step 2: Planner creates a plan
        plan = self.planner.chat(user_input)

        # Loop until the plan is approved by the Critic
        while True:
            # Step 3: Critic reviews the plan
            critique = self.critic.chat(plan)

            # Check if Critic approves the plan
            if "TERMINATE" in critique.upper():
                print("Plan approved by Critic.")
                break
            else:
                # Critic provides feedback, Planner revises the plan
                print("Critic feedback:")
                print(critique)
                plan = self.planner.chat(critique)

        # Step 4: UserProxyAgent approves the plan
        approval = self.user_proxy.chat(plan)

        # Check if UserProxyAgent approves the plan
        if "TERMINATE" not in approval.upper():
            print("User did not approve the plan. Exiting.")
            return

        # Step 5: Assistant executes the plan
        assistant_response = self.assistant.chat(plan)

        # Step 6: Coder generates code if needed
        if assistant_response and "CODE_NEEDED" in assistant_response.upper():
            code = self.coder.chat(assistant_response)

            # Step 7: Executor executes the code
            execution_result = self.executor.chat(code)

            # Loop to handle code execution errors
            while True:
                if "Error" in execution_result:
                    # Executor reports an error, Coder fixes the code
                    print("Execution error:")
                    print(execution_result)
                    code = self.coder.chat(execution_result)
                    execution_result = self.executor.chat(code)
                else:
                    # Execution successful
                    break

            # Step 8: Assistant processes the execution result
            assistant_result = self.assistant.chat(execution_result)
        else:
            assistant_result = assistant_response

        # Step 9: Return the final result
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
