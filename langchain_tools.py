from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from typing import List

from adb_controller import AdbController

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
        """Executes a system command like sleep, reboot, or poweroff, with user confirmation."""
        confirm = input(f"Confirm system command '{command}'? (y/n): ")
        if confirm.lower() == 'y':
            adb_controller.system_command(command)
            return f"Executed system command: {command}."
        return "System command aborted by user."

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
    ]

# The Python REPL tool does not depend on the ADB controller, so it can be defined separately.
python_repl_tool = PythonREPLTool()
