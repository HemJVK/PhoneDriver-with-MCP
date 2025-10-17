import os
import base64
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq

class GroqAgent:
    """
    An agent that uses Groq's cloud-based LLM to decide on actions
    based on a screenshot and a user request.
    """

    def __init__(self, api_key: str, model_name: str, tools: List[BaseTool]):
        """
        Initializes the GroqAgent.

        Args:
            api_key: The Groq API key.
            model_name: The name of the model to use (e.g., 'llama3-70b-8192').
            tools: A list of BaseTool instances for the agent to use.
        """
        if not api_key:
            raise ValueError("Groq API key is required.")

        self.llm = ChatGroq(
            temperature=0,
            model_name=model_name,
            api_key=api_key
        )
        self.tools = tools
        self.agent_executor = self._create_agent_executor()
        logging.info(f"GroqAgent initialized with model: {model_name}")

    def _create_agent_executor(self) -> AgentExecutor:
        """
        Creates the LangChain agent and executor.
        """
        system_prompt = """
        You are an expert at controlling a smartphone.
        You are given a user's request and a screenshot of the current screen.
        Your goal is to decide the next action to take to fulfill the user's request.
        Analyze the screenshot and the request carefully.
        Choose one of the available tools to perform the next action.
        If the task is complete, use the 'terminate' tool.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Encodes an image file to a base64 string.

        Args:
            image_path: The path to the image file.

        Returns:
            The base64-encoded image string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to encode image at {image_path}: {e}")
            raise

    def analyze_screenshot(
        self,
        screenshot_path: str,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes a screenshot and decides on the next action.

        Args:
            screenshot_path: The path to the screenshot.
            user_request: The user's high-level task request.
            context: The current session context, including previous actions.

        Returns:
            A dictionary representing the tool call to execute, or None if an error occurs.
        """
        if not os.path.exists(screenshot_path):
            logging.error(f"Screenshot not found at: {screenshot_path}")
            return None

        base64_image = self._encode_image(screenshot_path)

        history_str = "No previous actions."
        if context and context.get('previous_actions'):
            actions = [f"- {a['action']}" for a in context['previous_actions']]
            history_str = "Previous actions:\n" + "\n".join(actions)

        content = [
            {
                "type": "text",
                "text": f"User Request: {user_request}\n\n{history_str}\n\nAnalyze the screen and decide the next action."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        messages = [HumanMessage(content=content)]

        try:
            logging.info("Invoking Groq agent...")
            result = self.agent_executor.invoke({"messages": messages})

            # The output from a tool-calling agent is a list of tool calls
            if "output" in result and isinstance(result["output"], str):
                # This could be the direct output of the tool
                # We need to find which tool was called from the intermediate steps
                tool_calls = result.get("intermediate_steps", [])
                if tool_calls:
                    # Get the last tool call
                    last_tool_call = tool_calls[-1][0]
                    action = {
                        "action": last_tool_call.tool,
                        "args": last_tool_call.tool_input,
                        "reasoning": result.get("output", "No reasoning provided.")
                    }
                    return action

            logging.warning("Agent did not produce a clear tool call.")
            return None

        except Exception as e:
            logging.error(f"Error invoking Groq agent: {e}", exc_info=True)
            return None