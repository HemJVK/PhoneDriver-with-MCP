import os
import base64
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

class GroqAgent:
    """
    An agent that uses Groq's cloud-based LLM and LangGraph to decide on actions
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
        self.agent_executor = create_react_agent(self.llm, self.tools)
        logging.info(f"GroqAgent initialized with model: {model_name} and LangGraph.")

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
        Analyzes a screenshot and decides on the next action using the LangGraph agent.

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
            actions = [f"- {a['action']}({a.get('args', '')})" for a in context['previous_actions']]
            history_str = "Previous actions:\n" + "\n".join(actions)

        prompt_text = f"""
        You are an expert at controlling a smartphone.
        Your goal is to fulfill the user's request: "{user_request}".

        {history_str}

        Analyze the current screenshot and decide the single next best action to take.
        Choose one of the available tools. If the task is complete, use the 'terminate' tool.
        """

        content = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]

        messages = [HumanMessage(content=content)]

        try:
            logging.info("Invoking LangGraph agent...")
            # The create_react_agent expects a dictionary with a 'messages' key
            inputs = {"messages": messages}

            action_to_return = None
            reasoning = "No reasoning captured."

            # The default stream mode gives events as they happen from each node
            for event in self.agent_executor.stream(inputs):
                # The 'agent' node is where the LLM makes its decision
                if "agent" in event:
                    agent_output = event["agent"]
                    # The agent's output is a dictionary, typically with a 'messages' key
                    if "messages" in agent_output:
                        # The message from the agent contains the thought and the action
                        for message in agent_output["messages"]:
                            if message.content:
                                reasoning = message.content.strip()

                            if message.tool_calls:
                                # Get the first tool call
                                tool_call = message.tool_calls[0]
                                action_to_return = {
                                    "action": tool_call['name'],
                                    "args": tool_call['args'],
                                    "reasoning": reasoning
                                }
                                # We've found the action, we can stop processing this event
                                break

                # If we've found an action, we can exit the stream
                if action_to_return:
                    break

            if action_to_return:
                logging.info(f"Agent Reasoning: {action_to_return['reasoning']}")
                return action_to_return
            else:
                logging.warning("Agent did not produce a tool call.")
                return None

        except Exception as e:
            logging.error(f"Error invoking LangGraph agent: {e}", exc_info=True)
            return None