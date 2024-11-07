



###############################################################################################


# # # Load the indexed data using llama_index for RAG
# # DATA_DIR = os.getenv('DATA_DIR', 'data/antibody_antigen_models')
# # documents = SimpleDirectoryReader(DATA_DIR).load_data()
# # index = VectorStoreIndex(documents)
# # query_engine = index.as_query_engine()

# # Termination message function
# def termination_msg(x):
#     return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# # Custom UserProxyAgent with extended functionality
# class CustomUserProxyAgent(UserProxyAgent):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def chat(self, message):
#         print(f"UserProxyAgent received a message: {message}")
#         if "Please provide" in message:
#             # Prompt the user for input
#             user_input = input(message + "\n")
#             return user_input
#         else:
#             # Interact with the user to handle open-ended questions
#             if self.human_input_mode == "ALWAYS":
#                 user_response = input("User Input: ")
#                 return user_response
#             else:
#                 return super().chat(message)

# # Instantiate the custom UserProxyAgent
# user_proxy = CustomUserProxyAgent(
#     name="UserProxyAgent",
#     is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
#     human_input_mode="ALWAYS",
#     system_message=(
#         "You are the UserProxyAgent, representing the user in the system. "
#         "You interact with other agents by providing input, approving plans, and giving feedback. "
#         "Ensure that the agents' actions align with the user's goals and preferences. "
#         "Be proactive in requesting clarifications or additional information when needed."
#     ),
#     max_consecutive_auto_reply=None,
#     code_execution_config=False,
#     llm_config=llm_config,
# )

# # Coder Agent (unchanged)
# coder = AssistantAgent(
#     name="Coder",
#     system_message=(
#         "You are the Coder Agent. Your role is to write high-quality Python code to perform antibody design tasks as per the plan. "
#         "Understand the requirements, write efficient and correct code, and be ready to debug and fix issues based on feedback from the Executor. "
#         "Provide code in properly formatted code blocks, and ensure your code is executable and well-documented."
#     ),
#     llm_config=llm_config,
# )

# # Critic Agent (unchanged)
# critic = AssistantAgent(
#     name="Critic",
#     system_message=(
#         "You are the Critic Agent. Your role is to critically analyze the plans developed by the Planner. "
#         "Check for completeness, correctness, and feasibility, focusing on the functions and parameters used. "
#         "Provide constructive feedback to improve the plan, and ensure that it meets the objectives. "
#         "When the plan is satisfactory, indicate approval by stating 'TERMINATE'."
#     ),
#     llm_config=llm_config,
# )

# # Executor Agent (unchanged)
# executor = UserProxyAgent(
#     name="Executor",
#     system_message=(
#         "You are the Executor Agent. Your role is to execute the code provided by the Coder. "
#         "Run the code in a safe and controlled environment, capture any outputs or errors, and report back the results. "
#         "Ensure that the execution is secure and does not violate any policies. "
#         "Provide clear and detailed feedback on the execution outcome."
#     ),
#     human_input_mode="NEVER",
#     code_execution_config={
#         "last_n_messages": 12, 
#         "work_dir": "./code_antibody/",
#         "use_docker": False # not using docker for now
#         }, 
#     llm_config=llm_config,
# )

# # Planner Agent
# planner = AssistantAgent(
#     name="Planner",
#     system_message=(
#         "You are the Planner Agent. Your role is to develop a detailed plan to achieve the user's objectives in antibody discovery. "
#         "Begin by understanding the user's input and requirements. "
#         "Outline the steps needed, including identifying the necessary input parameters (antigen_sequence, origin_seq, origin_light, cdrh3_begin, cdrh3_end), "
#         "specifying that the Assistant should call the 'generate_antibody_sequence_palm_h3' function, "
#         "and then call 'analyze_antibody_properties' with the generated sequences. "
#         "Do not include optimization steps in the plan." # for now
#         "Ensure the plan is clear, actionable, and aligned with the goals."
#     ),
#     llm_config=llm_config,
# )

# # RetrieveUserProxyAgent (for RAG) (unchanged)
# # ragproxyagent = RetrieveUserProxyAgent(
# #     name="RetrieveUserProxyAgent",
# #     system_message=(
# #         "You are the RetrieveUserProxyAgent with advanced content retrieval capabilities for antibody domain knowledge. "
# #         "Your role is to provide relevant information from the knowledge base to assist other agents. "
# #         "Follow the plan and respond to queries with precise and accurate information."
# #     ),
# #     human_input_mode="NEVER",
# #     is_termination_msg=termination_msg,
# #     max_consecutive_auto_reply=10,
# #     retrieve_config={
# #         "task": "qa",
# #         "docs_path": f"{DATA_DIR}",
# #         "chunk_token_size": 3000,
# #         "model": llm_config['config_list'][0]["model"],
# #         "chunk_mode": "one_line",
# #         "embedding_model": "all-MiniLM-L6-v2",
# #         "get_or_create": True
# #     },
# #     llm_config=llm_config,
# # )

# # Custom AssistantAgent with extended functionality
# class CustomAssistantAgent(AssistantAgent):
#     def __init__(self, user_proxy_agent, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.user_proxy_agent = user_proxy_agent
#         self.generated_sequences = None  # Store generated sequences

#     def step(self, message):
#         # Check if the message indicates a need to call a function
#         if "generate_antibody_sequence_palm_h3" in message:
#             # Extract parameters from the message
#             try:
#                 import re
#                 params_str = re.search(r'generate_antibody_sequence_palm_h3\((.*)\)', message).group(1)
#                 params = json.loads(params_str)
#             except Exception as e:
#                 # If parsing fails, request the parameters from the user
#                 print("Assistant: Could not extract parameters. Requesting inputs from the user.")
#                 antigen_sequence = self.user_proxy_agent.chat("Please provide the antigen sequence:")
#                 origin_seq = self.user_proxy_agent.chat("Please provide the origin heavy chain sequence:")
#                 origin_light = self.user_proxy_agent.chat("Please provide the origin light chain sequence:")
#                 cdrh3_begin = int(self.user_proxy_agent.chat("Please provide the CDR H3 begin index (integer):"))
#                 cdrh3_end = int(self.user_proxy_agent.chat("Please provide the CDR H3 end index (integer):"))
#                 params = {
#                     "antigen_sequence": antigen_sequence,
#                     "origin_seq": origin_seq,
#                     "origin_light": origin_light,
#                     "cdrh3_begin": cdrh3_begin,
#                     "cdrh3_end": cdrh3_end
#                 }
#             # Call the function
#             df_results = self.function_map["generate_antibody_sequence_palm_h3"](**params)
#             self.generated_sequences = df_results  # Store the DataFrame

#             # Process and return the results
#             response_content = "Generated antibody sequences."
#             # response_content = f"Generated antibody sequences:\n{df_results.to_string()}"
#             return {"content": response_content, "role": "assistant"}

#         elif "analyze_antibody_properties" in message:
#             if self.generated_sequences is not None:
#                 analysis_reports = []
#                 for idx, row in self.generated_sequences.iterrows():
#                     heavy_chain = row['Heavy_Chain']
#                     light_chain = row['Light_Chain']
#                     # Call the analysis function
#                     analysis_result = self.function_map["analyze_antibody_properties"](
#                         heavy_chain=heavy_chain,
#                         light_chain=light_chain
#                     )
#                     analysis_reports.append(f"Sequence {idx+1}:\n{analysis_result}")
#                 # Return combined analysis reports
#                 response_content = "\n\n".join(analysis_reports)
#                 # response_content = f"Analysis Reports:\n\n{ '\n\n'.join(analysis_reports) }"
#                 return {"content": response_content, "role": "assistant"}
#             else:
#                 response_content = "No generated sequences available to analyze."
#                 return {"content": response_content, "role": "assistant"}

#         else:
#             # Default behavior: use the base class's step method
#             return super().step(message)

# # Instantiate the custom AssistantAgent
# assistant = CustomAssistantAgent(
#     user_proxy_agent=user_proxy,
#     name="Assistant",
#     system_message=(
#         "You are the Assistant Agent with access to custom functions for antibody discovery. "
#         "Your role is to execute functions as suggested by the Planner, prepare the required input parameters, and process the results. "
#         "When calling 'generate_antibody_sequence_palm_h3' and 'analyze_antibody_properties', ensure you provide all necessary parameters. "
#         "Do not execute optimization functions at this time." # for now
#         "Collaborate effectively with other agents, share insights, and contribute to achieving the overall objectives."
#     ),
#     llm_config=llm_config,
#     function_map={
#         "generate_antibody_sequence_palm_h3": func.generate_antibody_sequence_palm_h3,
#         "analyze_antibody_properties": func.analyze_antibody_properties,
#         # optimize coming soon...
#     },
# )


# # Coordinator class to manage agent interactions
# class Coordinator:
#     def __init__(self, user_proxy, planner, coder, critic, executor, assistant): # ragproxyagent
#         self.user_proxy = user_proxy
#         self.planner = planner
#         self.coder = coder
#         self.critic = critic
#         self.executor = executor
#         self.assistant = assistant
#         # self.ragproxyagent = ragproxyagent

#     def run(self):
#         # Step 1: Get user input
#         print("Welcome to the AbAgent for Antibody Discovery.")
#         print("Please describe your objective:")
#         user_input = self.user_proxy.chat("")

#         # Step 2: Planner creates a plan
#         # plan = self.planner.chat(user_input)
#         plan_response = self.planner.step(user_input)
#         plan = plan_response["content"]

#         # Loop until the plan is approved by the Critic
#         while True:
#             # Step 3: Critic reviews the plan
#             critique_response = self.critic.step(plan)
#             critique = critique_response["content"]

#             # Check if Critic approves the plan
#             if "TERMINATE" in critique.upper():
#                 print("Plan approved by Critic.")
#                 break
#             else:
#                 # critic provides feedback, planner revises plan
#                 print("Critic feedback:")
#                 print(critique)
#                 plan_response = self.planner.step(critique)
#                 plan = plan_response["content"]

#         # Step 4: UserProxyAgent approves the plan
#         print("Proposed Plan:")
#         print(plan)
#         approval = self.user_proxy.chat("Do you approve this plan? (yes/no)")

#         # Check if UserProxyAgent approves the plan
#         if "yes" not in approval.lower():
#             print("User did not approve the plan. Exiting.")
#             return

#         # Step 5: Assistant executes the plan
#         assistant_response = self.assistant.step(plan)
#         assistant_output = assistant_response["content"]

#         # Step 6: Handle Assistant's response
#         print("Assistant Response:")
#         print(assistant_response)

#         # Additional steps can be added here for code generation and execution if needed

#         # Step 9: Return the final result
#         print("Antibody Design Output:")
#         print(assistant_response)

# # Main execution
# def main():
#     coordinator = Coordinator(
#         user_proxy=user_proxy,
#         planner=planner,
#         coder=coder,
#         critic=critic,
#         executor=executor,
#         assistant=assistant,
#         # ragproxyagent=ragproxyagent,
#     )
#     coordinator.run()

# if __name__ == "__main__":
#     main()