import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from typing import List
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage

from adb_controller import AdbController
from langchain_tools import get_adb_tools

class GroqAgent:
    """
    A command-based agent that uses Groq's language model to interact with a phone.
    """

    def __init__(self, api_key: str, device_id: str = None, model_name: str = "llama3-70b-8192"):
        self.logger = logging.getLogger(__name__)
        self.adb_controller = AdbController(device_id)

        self.model = ChatGroq(api_key=api_key, model_name=model_name)

        self.tools = self._load_tools()
        self.agent_executor = self._create_agent_executor()

    def _load_tools(self) -> List[Tool]:
        """
        Loads the tools that the agent can use using the factory function.
        """
        return get_adb_tools(self.adb_controller)

    def _create_agent_executor(self):
        """
        Creates the agent executor using LangGraph.
        """
        return create_react_agent(self.model, self.tools)

    def run_task(self, user_request: str):
        """
        Runs the agent to complete a task based on text and tool outputs only.
        """
        self.logger.info(f"Starting pure text-based task: {user_request}")

        system_prompt = (
            "You are an expert phone automation assistant. Your only purpose is to execute tasks on a phone based on the user's request. "
            "You will be given a user's high-level goal. Your response MUST be a single, valid, properly formatted tool call to accomplish the next step. "
            "Do NOT provide any conversational text, explanations, or additional commentary. Your output must be ONLY the tool call."
            "The screen resolution is {width}x{height}. All coordinates must be within these bounds."
        ).format(width=self.adb_controller.width, height=self.adb_controller.height)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_request)
        ]

        response = self.agent_executor.invoke({"messages": messages})

        final_output = response["messages"][-1].content
        self.logger.info(f"Task finished. Final output: {final_output}")
        return final_output
