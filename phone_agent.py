import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from adb_mcp_client import AdbMcpClient
from groq_agent import GroqAgent
from langchain_tools import TapTool, SwipeTool, TypeTool, WaitTool, TerminateTool
from langchain.tools import BaseTool

class PhoneAgent:
    """
    Orchestrates the phone automation workflow using a Groq-powered LangChain agent
    and an ADB MCP client.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the PhoneAgent.

        Args:
            config: A dictionary with configuration parameters.
        """
        default_config = {
            "screenshot_dir": "./screenshots",
            "max_retries": 3,
            "step_delay": 1.5,
            "adb_server_address": "http://localhost:8080",
            "groq_api_key": os.environ.get("GROQ_API_KEY"),
            "groq_model_name": "llama3-70b-8192",
        }
        self.config = default_config
        if config:
            self.config.update(config)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logging()
        self._setup_directories()

        self.adb_client = AdbMcpClient(self.config["adb_server_address"])
        self.tools = self._initialize_tools()
        self.groq_agent = GroqAgent(
            api_key=self.config["groq_api_key"],
            model_name=self.config["groq_model_name"],
            tools=self.tools
        )

        self.context: Dict[str, Any] = {
            "previous_actions": [],
            "task_request": "",
            "session_id": self.session_id,
            "screenshots": [],
        }
        logging.info("PhoneAgent is ready.")

    def _setup_logging(self):
        log_file = f"phone_agent_{self.session_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        logging.info(f"Session started: {self.session_id}")

    def _setup_directories(self):
        Path(self.config["screenshot_dir"]).mkdir(parents=True, exist_ok=True)

    def _initialize_tools(self) -> List[BaseTool]:
        """Initializes the custom tools with the ADB client."""
        return [
            TapTool(adb_client=self.adb_client),
            SwipeTool(adb_client=self.adb_client),
            TypeTool(adb_client=self.adb_client),
            WaitTool(),
            TerminateTool(),
        ]

    def execute_cycle(self, user_request: str) -> Dict[str, Any]:
        """
        Executes a single interaction cycle.

        Args:
            user_request: The user's task request.

        Returns:
            A dictionary with the result of the cycle.
        """
        try:
            screenshot_path = self.adb_client.capture_screenshot()
            self.context["screenshots"].append(screenshot_path)

            action = self.groq_agent.analyze_screenshot(
                screenshot_path, user_request, self.context
            )

            if not action:
                raise Exception("Failed to get a valid action from the agent.")

            logging.info(f"Agent decided to perform: {action['action']} with args: {action.get('args')}")
            logging.info(f"Reasoning: {action.get('reasoning')}")

            tool_to_execute = next((t for t in self.tools if t.name == action["action"]), None)
            if not tool_to_execute:
                raise ValueError(f"Unknown action type: {action['action']}")

            result = tool_to_execute.run(action["args"])
            logging.info(f"Tool execution result: {result}")

            self.context["previous_actions"].append({
                "action": action["action"],
                "args": action.get("args"),
                "timestamp": time.time(),
            })

            time.sleep(self.config["step_delay"])

            return {
                "success": True,
                "action": action,
                "result": result,
                "task_complete": action["action"] == "terminate",
            }

        except Exception as e:
            logging.error(f"Cycle execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def execute_task(self, user_request: str, max_cycles: int = 15) -> Dict[str, Any]:
        """
        Executes a complete task over multiple cycles.

        Args:
            user_request: The user's task description.
            max_cycles: The maximum number of cycles to run.

        Returns:
            A dictionary summarizing the task execution.
        """
        self.context["task_request"] = user_request
        logging.info(f"STARTING TASK: {user_request}")

        for cycle_num in range(1, max_cycles + 1):
            logging.info(f"\n--- Cycle {cycle_num}/{max_cycles} ---")
            result = self.execute_cycle(user_request)

            if not result["success"]:
                logging.error(f"Task failed due to an error: {result.get('error')}")
                return {"success": False, "cycles": cycle_num, "error": result.get("error")}

            if result.get("task_complete"):
                logging.info(f"Task completed successfully: {result.get('result')}")
                return {"success": True, "cycles": cycle_num, "result": result.get("result")}

        logging.warning("Task incomplete: reached max cycles.")
        return {"success": False, "cycles": max_cycles, "reason": "Max cycles reached."}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phone_agent.py 'your task here'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    config_path = "config.json"
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    # Ensure the API key is set, either from env or config
    if "groq_api_key" not in config and "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY is not set. Please set it as an environment variable or in config.json.")
        sys.exit(1)

    agent = PhoneAgent(config)
    result = agent.execute_task(task)

    if result["success"]:
        print(f"\nTask completed in {result['cycles']} cycles.")
    else:
        print(f"\nTask failed after {result['cycles']} cycles.")