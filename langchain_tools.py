import logging
import time
from typing import Type, Any, Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class TapToolInput(BaseModel):
    x: int = Field(description="The x-coordinate to tap.")
    y: int = Field(description="The y-coordinate to tap.")

class TapTool(BaseTool):
    name = "tap"
    description = "Taps a specific coordinate on the screen."
    args_schema: Type[BaseModel] = TapToolInput
    adb_client: AdbMcpClient

    def _run(self, x: int, y: int) -> str:
        try:
            self.adb_client.tap(x, y)
            return f"Tapped at ({x}, {y})."
        except Exception as e:
            logging.error(f"Error in TapTool: {e}")
            return f"Failed to tap at ({x}, {y})."

class SwipeToolInput(BaseModel):
    start_x: int = Field(description="The starting x-coordinate of the swipe.")
    start_y: int = Field(description="The starting y-coordinate of the swipe.")
    end_x: int = Field(description="The ending x-coordinate of the swipe.")
    end_y: int = Field(description="The ending y-coordinate of the swipe.")
    duration: int = Field(description="The duration of the swipe in milliseconds.")

class SwipeTool(BaseTool):
    name = "swipe"
    description = "Swipes from a starting point to an ending point on the screen."
    args_schema: Type[BaseModel] = SwipeToolInput
    adb_client: AdbMcpClient

    def _run(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int) -> str:
        try:
            self.adb_client.swipe(start_x, start_y, end_x, end_y, duration)
            return f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})."
        except Exception as e:
            logging.error(f"Error in SwipeTool: {e}")
            return f"Failed to swipe."

class TypeToolInput(BaseModel):
    text: str = Field(description="The text to type.")

class TypeTool(BaseTool):
    name = "type"
    description = "Types the given text into the currently focused input field."
    args_schema: Type[BaseModel] = TypeToolInput
    adb_client: AdbMcpClient

    def _run(self, text: str) -> str:
        try:
            self.adb_client.type_text(text)
            return f"Typed: '{text}'."
        except Exception as e:
            logging.error(f"Error in TypeTool: {e}")
            return f"Failed to type '{text}'."

class WaitToolInput(BaseModel):
    milliseconds: int = Field(description="The number of milliseconds to wait.")

class WaitTool(BaseTool):
    name = "wait"
    description = "Waits for a specified amount of time."
    args_schema: Type[BaseModel] = WaitToolInput

    def _run(self, milliseconds: int) -> str:
        wait_time_seconds = milliseconds / 1000.0
        time.sleep(wait_time_seconds)
        return f"Waited for {milliseconds} milliseconds."

class TerminateToolInput(BaseModel):
    message: str = Field(description="The final message to report.")

class TerminateTool(BaseTool):
    name = "terminate"
    description = "Terminates the task and provides a final message."
    args_schema: Type[BaseModel] = TerminateToolInput

    def _run(self, message: str) -> str:
        return f"Task terminated with message: {message}"