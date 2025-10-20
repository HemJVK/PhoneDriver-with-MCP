import subprocess
import sys
import logging
from langchain_core.tools import tool
from typing import List

from adb_controller import AdbController

@tool
def finish_task(reason: str) -> str:
    """
    Call this tool to signal that the user's task is complete.
    Provide a brief reason explaining why you believe the task is finished.
    """
    return f"Task finished: {reason}"

@tool
def python_repl(code: str) -> str:
    """
    A Python REPL tool. Use this to execute python code.
    Input should be a valid python code string.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, timeout=30
        )
        return f"Execution successful:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except Exception as e:
        return f"An error occurred: {e}"

def get_adb_tools(adb_controller: AdbController) -> List:
    """Factory function to create ADB tools with a given controller."""

    @tool
    def tap(x: int, y: int) -> str:
        """Tap on a specific coordinate on the screen."""
        adb_controller.tap(x, y)
        return f"Tapped screen at ({x}, {y})."

    @tool
    def swipe(direction: str) -> str:
        """Swipe on the screen in a given direction (up, down, left, right)."""
        adb_controller.swipe(direction)
        return f"Swiped {direction}."

    @tool
    def type_text(text: str) -> str:
        """Type text into a text field."""
        adb_controller.type_text(text)
        return f"Typed '{text}'."

    @tool
    def open_notification_shade() -> str:
        """Opens the notification shade on the device."""
        adb_controller.open_notifications()
        return "Opened notification shade."

    @tool
    def make_phone_call(number: str) -> str:
        """Makes a phone call to a specified number."""
        adb_controller.make_call(number)
        return f"Made a call to {number}."

    @tool
    def send_message(number: str, message: str) -> str:
        """Sends a text message to a specified number."""
        adb_controller.send_message(number, message)
        return f"Sent message to {number}."

    @tool
    def system_command(command: str) -> str:
        """Executes a system command like sleep, reboot, or poweroff. USE WITH CAUTION."""
        logging.warning(f"Executing potentially disruptive system command: '{command}'.")
        adb_controller.system_command(command)
        return f"Executed system command: {command}."

    @tool
    def take_photo() -> str:
        """Takes a photo using the camera."""
        adb_controller.take_photo()
        return "Took a photo."

    return [
        tap,
        swipe,
        type_text,
        open_notification_shade,
        make_phone_call,
        send_message,
        system_command,
        take_photo,
        python_repl,
        finish_task, # Add the new tool here
    ]
