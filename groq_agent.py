import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from typing import List
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage

from adb_controller import AdbController
from langchain_tools import get_adb_tools
from vision_analyzer import VisionAnalyzer

class GroqAgent:
    """
    An agent that uses a dual-LLM approach: a Groq vision model to see and a Groq text model to reason and act.
    """

    def __init__(self, api_key: str, device_id: str = None, text_model: str = "mixtral-8x7b-32768", vision_model: str = "l4-scout-17b"):
        self.logger = logging.getLogger(__name__)
        self.adb_controller = AdbController(device_id)
        self.vision_analyzer = VisionAnalyzer(api_key=api_key, model_name=vision_model)

        # The reasoning model
        self.model = ChatGroq(api_key=api_key, model_name=text_model, temperature=0)

        self.tools = self._load_tools()
        self.agent_executor = self._create_agent_executor()

    def _load_tools(self) -> List[Tool]:
        """Loads the tools that the agent can use."""
        return get_adb_tools(self.adb_controller)

    def _create_agent_executor(self):
        """Creates the agent executor using LangGraph."""
        return create_react_agent(self.model, self.tools)

    def run_task_loop(self, user_request: str, max_steps: int = 10):
        """
        Runs the agent in an iterative "observe-think-act" loop to complete a task.
        """
        self.logger.info(f"Starting task with dual-LLM loop: {user_request}")

        system_prompt = (
            "You are a precise phone automation assistant. Your only purpose is to execute tasks on a phone. "
            "You will be given a user's high-level goal and a description of the current screen. "
            "Your response MUST be a single, valid, properly formatted tool call to accomplish the next step. "
            "If you believe the task is complete, use the 'finish_task' tool. "
            "Do NOT provide conversational text or explanations. Your output must be ONLY the tool call. "
            "Pay extremely close attention to the tool's schema. For `tap`, `x` and `y` must be integers. "
            "The screen resolution is {width}x{height}."
        ).format(width=self.adb_controller.width, height=self.adb_controller.height)

        for step in range(max_steps):
            self.logger.info(f"--- Step {step + 1}/{max_steps} ---")

            # 1. OBSERVE: Look at the screen
            screenshot_path = self.adb_controller.capture_screenshot()
            screen_description = self.vision_analyzer.describe_screen(screenshot_path)

            # 2. THINK: Decide on the next action
            reasoning_prompt = (
                f"**User's Goal:** {user_request}\n\n"
                f"**Current Screen Description:**\n{screen_description}\n\n"
                f"Based on the user's goal and the current screen, what is the single best tool call to make next? "
                "Remember, your response must be only the tool call. Use 'finish_task' if the goal is met."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=reasoning_prompt)
            ]

            response = self.agent_executor.invoke({"messages": messages})

            # 3. ACT: Execute the action
            # The agent's response is the action, which is automatically executed by the LangGraph agent.
            # We just need to check if the task is finished.
            last_message = response["messages"][-1]
            if "finish_task" in str(last_message.content):
                self.logger.info("Agent decided the task is finished.")
                return "Task finished successfully."

        self.logger.warning("Max steps reached. Task may be incomplete.")
        return "Task finished due to reaching max steps."
