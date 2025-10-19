import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.tools import Tool

from adb_controller import AdbController
from langchain_tools import get_adb_tools

class GroqAgent:
    """
    A command-based agent that uses Groq's language model to interact with a phone.
    """

    def __init__(self, api_key: str, device_id: str = None, model_name: str = "llama3-70b-8192"):
        self.logger = logging.getLogger(__name__)
        self.model = ChatGroq(api_key=api_key, model_name=model_name)
        self.adb_controller = AdbController(device_id)
        self.tools = self._load_tools()
        self.agent_executor = self._create_agent_executor()

    def _load_tools(self) -> List[Tool]:
        """
        Loads the tools that the agent can use using the factory function.
        """
        # The factory function now returns all tools, including the python_repl.
        return get_adb_tools(self.adb_controller)

    def _create_agent_executor(self):
        """
        Creates the agent executor using LangGraph, including a custom system prompt.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that can control a phone. "
                    "You have access to a set of tools to interact with the device. "
                    "The screen resolution is {width}x{height}. "
                    "When using the 'tap' tool, you must provide coordinates within these bounds. "
                    "Always consider the user's request and the history of previous actions to decide on the next step. "
                    "When the task is complete, use your own words to summarize what you did and why."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        bound_prompt = prompt.partial(
            width=self.adb_controller.width,
            height=self.adb_controller.height
        )

        return create_react_agent(self.model, self.tools, messages_modifier=bound_prompt)

    def run_task(self, user_request: str):
        """
        Runs the agent to complete a task.
        """
        self.logger.info(f"Starting task: {user_request}")

        response = self.agent_executor.invoke({"messages": [("human", user_request)]})

        final_output = response["messages"][-1].content
        self.logger.info(f"Task finished. Final output: {final_output}")
        return final_output
