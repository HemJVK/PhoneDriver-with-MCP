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
    An agent that uses a dual-LLM approach: a vision model to see and a Groq text model to reason and act.
    Includes planning and human-in-the-loop review capabilities.
    """

    def __init__(self, api_key: str, device_id: str = None, text_model: str = "gemma-7b-it"):
        self.logger = logging.getLogger(__name__)
        self.adb_controller = AdbController(device_id)
        self.vision_analyzer = VisionAnalyzer()

        self.model = ChatGroq(api_key=api_key, model_name=text_model, temperature=0)

        self.tools = self._load_tools()
        self.agent_executor = self._create_agent_executor()

    def _load_tools(self) -> List[Tool]:
        """Loads the tools that the agent can use."""
        return get_adb_tools(self.adb_controller)

    def _create_agent_executor(self):
        """Creates the agent executor using LangGraph."""
        return create_react_agent(self.model, self.tools)

    def generate_plan(self, user_request: str) -> str:
        """
        Generates a high-level, step-by-step plan for the user to review.
        """
        self.logger.info("Generating a plan for the user's request...")

        screenshot_path = self.adb_controller.capture_screenshot()
        screen_description = self.vision_analyzer.describe_screen(screenshot_path)

        if "Error:" in screen_description:
            return "Could not generate a plan because the vision model failed to analyze the screen."

        prompt = (
            "You are a meticulous planning agent. Your task is to create a clear, step-by-step plan to achieve the user's goal on a smartphone. "
            "You will be given the user's goal and a description of the current screen. "
            "Break the task down into simple, high-level actions. For example: '1. Tap the Chrome icon. 2. Type 'weather' into the search bar. 3. Tap the search button.'\n\n"
            f"**User's Goal:** {user_request}\n\n"
            f"**Current Screen Description:**\n{screen_description}\n\n"
            "**Your Plan:**"
        )

        response = self.model.invoke(prompt)
        plan = response.content
        self.logger.info(f"Generated Plan:\n{plan}")
        return plan

    def run_task_loop(self, user_request: str, plan: str, max_steps: int = 15):
        """
        Runs the agent in an iterative "observe-think-act" loop, guided by a plan.
        """
        self.logger.info(f"Executing task with plan: {user_request}")

        action_history = []

        system_prompt = (
            "You are a precise phone automation assistant. Your only purpose is to execute the next step of a given plan. "
            "You will be given the user's goal, the overall plan, a history of actions, and a description of the current screen. "
            "Your response MUST be a single, valid tool call to accomplish the *next* logical step. "
            "If the task is complete, use the 'finish_task' tool. Do NOT deviate from the plan."
        )

        for step in range(max_steps):
            self.logger.info(f"--- Step {step + 1}/{max_steps} ---")

            screenshot_path = self.adb_controller.capture_screenshot()
            screen_description = self.vision_analyzer.describe_screen(screenshot_path)

            if "Error:" in screen_description:
                self.logger.error("Vision model failed during execution loop. Aborting task.")
                return "Task failed because the vision model could not analyze the screen."

            history_str = "\n".join(action_history) if action_history else "No actions taken yet."

            reasoning_prompt = (
                f"**User's Goal:** {user_request}\n\n"
                f"**Overall Plan:**\n{plan}\n\n"
                f"**Action History:**\n{history_str}\n\n"
                f"**Current Screen Description:**\n{screen_description}\n\n"
                f"Based on the plan, history, and screen, what is the single best tool call for the next step? "
                "Use 'finish_task' if the plan is complete."
            )

            messages = [ SystemMessage(content=system_prompt), HumanMessage(content=reasoning_prompt) ]

            response = self.agent_executor.invoke({"messages": messages})

            last_message = response["messages"][-1]
            action_representation = str(last_message.content)
            action_history.append(f"Step {step + 1}: {action_representation}")

            if "finish_task" in action_representation:
                self.logger.info("Agent decided the task is finished.")
                return "Task finished successfully."

        self.logger.warning("Max steps reached. Task may be incomplete.")
        return "Task finished due to reaching max steps."
