'''
example code from LLM Agent Berkeley Lecture 3
- Multi-LLM framework using Autogen
'''
import autogen

writer = Writer("writer", llm_config=llm_config)
safeguard = autogen.AssistantAgent("safeguard", llm_config=llm_config)
commander = Commander("commander", llm_config=llm_config)
user = autogen.UserProxyAgent("user")