# import os
# import openai
# from autogen import AssistantAgent, UserProxyAgent
# from autogen.agentchat.groupchat import GroupChat, GroupChatManager
# from llm_config import immunologist_llm_config, project_lead_critic_llm_config, machine_learning_engineer_llm_config
# import agent_functions as func

# openai.api_key = os.getenv('OPENAI_API_KEY')

# user_proxy = UserProxyAgent(
#     name="UserProxyAgent",
#     system_message=(
#         "You are the UserProxyAgent, representing the user. "
#         "You pass the user's objectives and clarifications to the team. "
#         "You do not generate solutions yourself, just pass along user inputs."
#     ),
#     human_input_mode="ALWAYS",
#     llm_config=project_lead_critic_llm_config,
#     code_execution_config={"use_docker": False}
# )

# project_lead = AssistantAgent(
#     name="ProjectLead",
#     system_message=(
#         "You are the ProjectLead, facilitating a natural lab conversation:\n"
#         " - Let each participant speak in a realistic order.\n"
#         " - If the Critic or Immunologist spontaneously chime in, you allow it, as in a real meeting.\n"
#         " - However, keep the conversation on track to achieve the user's objective.\n"
#         " - Specifically ask each agent to clarify their role in turn: Critic, then MLE, then Immunologist.\n"
#         " - Finally, ask the Immunologist for the final plan.\n"
#         " - Then ask the Critic to review.\n"
#         " - After Critic says 'TERMINATE', ask the MLE to run `generate_ab` and `analyze_ab`.\n"
#         "You do not let the MLE skip clarifying its role. If the MLE tries to skip, politely request them to do so.\n"
#         "You remain in charge, but you're open to side comments from the team.\n"
#         "\n"
#         "After receiving the user's objective from the UserProxyAgent:\n"
#         "1. Start by asking each agent in turn (Critic, MachineLearningEngineer, Immunologist) to clarify their roles.\n"
#         "2. After roles are clarified, ask the Immunologist to provide:\n"
#         "   - A known antigen amino acid sequence as 'use_antigen'.\n"
#         "   - A known wild-type antibody heavy chain amino acid sequence ('origin_seq').\n"
#         "   - A known wild-type antibody light chain amino acid sequence ('origin_light').\n"
#         "   - Integer indices for CDRH3 (cdrh3_begin, cdrh3_end).\n"
#         "   Also ask for scientific/literature justification.\n"
#         "3. Once the Immunologist provides the final plan (all 5 parameters), ask the Critic to review.\n"
#         "4. The Critic must verify correctness and say 'TERMINATE' if correct.\n"
#         "5. After Critic approval, ask the MachineLearningEngineer to execute `generate_ab` and then `analyze_ab`.\n"
#         "Direct the conversation step-by-step so that each agent responds in turn.\n"
#     ),
#     llm_config=project_lead_critic_llm_config,
#     code_execution_config={"use_docker": False}
# )

# critic = AssistantAgent(
#     name="Critic",
#     system_message=(
#         "You are the Critic in a lab meeting.\n."
#         "You review the final plan provided by the Immunologist. If correct, say 'TERMINATE'.\n"
#         "You can chime in with feedback at any time, but do NOT lead the conversation or assign tasks.\n"
#         "Your domain is correctness checks, not conversation flow."
#         "\n"
#         "After the Immunologist provides the final plan:\n"
#         "1. Verify 'origin_seq', 'origin_light', 'use_antigen' are valid amino acid sequences.\n"
#         "2. Check 'cdrh3_begin' and 'cdrh3_end' are integers.\n"
#         "3. If correct, respond ONLY with 'TERMINATE' (no code blocks, no plan). "
#         "   If incorrect, provide specific feedback.\n"
#         "You do NOT provide the final plan yourself. You do NOT ask for user input.\n"
#         "When you're prompted by the ProjectLead, finalize your verdict."
#     ),
#     llm_config=project_lead_critic_llm_config,
#     code_execution_config={"use_docker": False}
# )

# machine_learning_engineer = AssistantAgent(
#     name="MachineLearningEngineer",
#     system_message=(
#         "You are the MachineLearningEngineer in a lab meeting.\n"
#         "You wait for natural openings in conversation or for the ProjectLead to address you.\n"
#         "You may jump in if it's directly relevant to your engineering tasks, but do NOT usurp the ProjectLead's role.\n"
#         "If someone else asks you a direct question, you answer. If the ProjectLead calls on you, you comply.\n"
#         "You DO NOT reorder the conversation, run the meeting, or assign tasks to others.\n"
#         "\n"
#         "When asked about your role, briefly clarify that you handle all script-based antibody generation (`generate_ab`) "
#         "and analysis (`analyze_ab`)."
#         "When asked by the ProjectLead (especially after Critic says 'TERMINATE'), you will:\n"
#         "1. Explain how you will run `generate_ab` and `analyze_ab`.\n"
#         "2. Then call `generate_ab` with the provided parameters.\n"
#         "3. After receiving generation results, call `analyze_ab`.\n"
#         "No user input requests needed from you. You respond when asked by ProjectLead."
#     ),
#     function_map={
#         "generate_ab": func.generate_ab,
#         "analyze_ab": func.analyze_ab, # calls analyze_ab(result_csv_path)
#     },
#     llm_config=machine_learning_engineer_llm_config,
#     code_execution_config={"use_docker": False}
# )

# immunologist = AssistantAgent(
#     name="Immunologist",
#     system_message=(
#         "You are the Immunologist. You can speak up in a meeting like a real colleague, \n"
#         "but your main job is providing the final plan: antigen sequence, heavy chain, light chain, etc.\n"
#         "You do not lead the conversation. If the ProjectLead or MLE or Critic ask you a question, answer it.\n"
#         "When the ProjectLead asks for the final plan, you will provide:\n"
#         "- 'use_antigen': a known antigen.\n"
#         "- 'origin_seq': a known wild-type antibody heavy chain sequence.\n"
#         "- 'origin_light': a known wild-type antibody light chain sequence.\n"
#         "- 'cdrh3_begin', 'cdrh3_end': integers for CDRH3 region.\n"
#         "All must be valid amino acids and no placeholders.\n"
#         "provide them in **plain text** (no triple backticks!).\n"
#         "Also provide scientific/literature justification.\n"
#         "You respond only when ProjectLead asks, then present these parameters clearly. (no code blocks) \n"
#     ),
#     llm_config=immunologist_llm_config,
#     code_execution_config={"use_docker": False}
# )

# def main():
#     from autogen.agentchat.groupchat import GroupChat, GroupChatManager

#     # Create a group chat with all agents
#     group_chat = GroupChat(
#         agents=[user_proxy, project_lead, critic, machine_learning_engineer, immunologist],
#         messages=[],
#         max_round=20,
#         speaker_selection_method="auto",
#         allow_repeat_speaker=True
#     )

#     chat_manager = GroupChatManager(group_chat)

#     user_input = input("Welcome to the Antibody Design Assistant.\nPlease describe your objective:\n")

#     # Start the conversation by having the UserProxyAgent relay the user's objective to the group
#     # The first message is from the user_proxy to the group chat manager
#     user_proxy.initiate_chat(
#         recipient=chat_manager,
#         message=user_input,
#         clear_history=False,
#     )

#     # Now run the chat. With speaker_selection_method="auto", the system will pick the next speaker.
#     # The system messages guide each agent on when to speak.
#     done, reason = chat_manager.run_chat()
#     if done:
#         print("Conversation ended:", reason)

# if __name__ == "__main__":
#     main()
import os
import openai
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from llm_config import immunologist_llm_config, project_lead_critic_llm_config, machine_learning_engineer_llm_config
import agent_functions as func

openai.api_key = os.getenv('OPENAI_API_KEY')

user_proxy = UserProxyAgent(
    name="UserProxyAgent",
    system_message=(
        "You are the UserProxyAgent, representing the user. "
        "You pass the user's objectives and clarifications to the team. "
        "You do not generate solutions yourself; just pass along user inputs."
    ),
    human_input_mode="ALWAYS",  # User can input after each message if needed
    llm_config=project_lead_critic_llm_config,
    code_execution_config={"use_docker": False}
)

project_lead = AssistantAgent(
    name="ProjectLead",
    system_message=(
        "You are the ProjectLead, facilitating a natural lab meeting about antibody design:\n"
        "- Let each participant speak in a free-flowing manner, BUT keep the meeting on track.\n"
        "- Specifically:\n"
        "   1) Ask Critic to clarify their role.\n"
        "   2) Ask MachineLearningEngineer (MLE) to clarify their role.\n"
        "   3) Ask Immunologist to clarify their role.\n"
        "   4) THEN request the Immunologist’s final plan: `use_antigen`, `origin_seq`, `origin_light`, "
        "      `cdrh3_begin`, `cdrh3_end`, plus justification.\n"
        "   5) Ask Critic to review and respond with 'TERMINATE' or feedback.\n"
        "   6) If Critic says 'TERMINATE', instruct MLE to run `generate_ab` then `analyze_ab`.\n"
        "\n"
        " - If any agent is explicitly addressed, do NOT let other agents overshadow them.\n"
        " - If any agent tries to skip clarifying or provide the plan, politely redirect them.\n"
        " - The final plan must have valid amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y) for sequences.\n"
        " - Keep it relaxed yet organized, so we achieve the user's objective.\n"
        "\n"
        "When you receive the user's objective from UserProxyAgent, greet the team and begin the steps above.\n"
    ),
    llm_config=project_lead_critic_llm_config,
    code_execution_config={"use_docker": False}
)

critic = AssistantAgent(
    name="Critic",
    system_message=(
        "You are the Critic in a lab meeting:\n"
        "- You ONLY verify the correctness of the Immunologist's final plan.\n"
        "- If the plan is correct (valid amino acids, correct indices) then respond with 'TERMINATE'.\n"
        "- If it's incorrect, provide feedback.\n"
        "- Do NOT produce the plan yourself.\n"
        "- Do NOT override the Immunologist or MLE.\n"
        "\n"
        "Additional detail:\n"
        "1) 'use_antigen', 'origin_seq', 'origin_light' must be valid amino acid sequences: only [ACDEFGHIKLMNPQRSTVWY].\n"
        "2) 'cdrh3_begin' and 'cdrh3_end' must be integers.\n"
        "3) If valid, respond with 'TERMINATE'. If not, specify what's wrong.\n"
        "No code blocks or user input requests.\n"
    ),
    llm_config=project_lead_critic_llm_config,
    code_execution_config={"use_docker": False}
)

machine_learning_engineer = AssistantAgent(
    name="MachineLearningEngineer",
    system_message=(
        "You are the MachineLearningEngineer (MLE) in a lab meeting:\n"
        "- You handle the script-based functions `generate_ab` and `analyze_ab`.\n"
        "- If the ProjectLead addresses you to clarify your role, mention that you:\n"
        "    * Generate antibody candidates from the given sequences.\n"
        "    * Analyze results afterwards.\n"
        "- If the ProjectLead specifically instructs you (after Critic says 'TERMINATE'), you:\n"
        "    1) Summarize how you will call `generate_ab` with the final plan.\n"
        "    2) Actually call `generate_ab`.\n"
        "    3) Then call `analyze_ab` with the CSV path.\n"
        "\n"
        "- Do NOT provide the Immunologist’s plan or override them.\n"
        "- Only speak when relevant to your tasks.\n"
    ),
    function_map={
        "generate_ab": func.generate_ab,
        "analyze_ab": func.analyze_ab,
    },
    llm_config=machine_learning_engineer_llm_config,
    code_execution_config={"use_docker": False}
)

immunologist = AssistantAgent(
    name="Immunologist",
    system_message=(
        "You are the Immunologist in a lab meeting:\n"
        "- You may spontaneously comment if needed, but do NOT overshadow the ProjectLead.\n"
        "- Specifically, when the ProjectLead asks for the final plan, you provide:\n"
        "   1) use_antigen (valid amino acid sequence),\n"
        "   2) origin_seq (valid amino acid sequence for heavy chain),\n"
        "   3) origin_light (valid amino acid sequence for light chain),\n"
        "   4) cdrh3_begin, cdrh3_end (integer indices),\n"
        "   plus a brief justification.\n"
        "- Provide these in plain text (no triple backticks). Only use uppercase letters A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y.\n"
        "- No placeholders, e.g. 'X' or 'Z'. Indices must be integers.\n"
        "- Do NOT lead the entire conversation. Only respond about the plan when the ProjectLead specifically asks.\n"
    ),
    llm_config=immunologist_llm_config,
    code_execution_config={"use_docker": False}
)

def main():
    from autogen.agentchat.groupchat import GroupChat, GroupChatManager

    # Create a group chat with all agents
    group_chat = GroupChat(
        agents=[
            user_proxy,
            project_lead,
            critic,
            machine_learning_engineer,
            immunologist
        ],
        messages=[],
        max_round=200,
        speaker_selection_method="auto",  # For a more natural flow
        allow_repeat_speaker=True
    )

    chat_manager = GroupChatManager(group_chat)

    user_input = input("Welcome to the Antibody Design Assistant.\nPlease describe your objective:\n")

    # The UserProxyAgent relays the user's objective as the first message.
    user_proxy.initiate_chat(
        recipient=chat_manager,
        message=user_input,
        clear_history=False,
    )

    # Now run the chat. The system messages instruct each agent how to speak/act.
    done, reason = chat_manager.run_chat()
    if done:
        print("Conversation ended:", reason)

if __name__ == "__main__":
    main()
