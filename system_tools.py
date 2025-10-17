import logging
from typing import Type
from pydantic import BaseModel
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class OpenNotificationShadeTool(BaseTool):
    name: str = "open_notification_shade"
    description: str = "Opens the notification shade from the top of the screen."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            self.adb_client.open_notifications()
            return "Successfully opened the notification shade."
        except Exception as e:
            logging.error(f"Error in OpenNotificationShadeTool: {e}")
            return "Failed to open the notification shade."

class OpenQuickSettingsTool(BaseTool):
    name: str = "open_quick_settings"
    description: str = "Opens the quick settings panel."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            self.adb_client.open_quick_settings()
            return "Successfully opened quick settings."
        except Exception as e:
            logging.error(f"Error in OpenQuickSettingsTool: {e}")
            return "Failed to open quick settings."

class SleepTool(BaseTool):
    name: str = "put_device_to_sleep"
    description: str = "Puts the device to sleep (turns the screen off)."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            self.adb_client.sleep()
            return "Device is now asleep."
        except Exception as e:
            logging.error(f"Error in SleepTool: {e}")
            return "Failed to put the device to sleep."

class RebootTool(BaseTool):
    name: str = "reboot_device"
    description: str = "Reboots the device. This is a dangerous action and requires user confirmation."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            # The confirmation logic will be in the agent, not the tool itself.
            self.adb_client.reboot()
            return "Device is rebooting."
        except Exception as e:
            logging.error(f"Error in RebootTool: {e}")
            return "Failed to reboot the device."

class PoweroffTool(BaseTool):
    name: str = "poweroff_device"
    description: str = "Powers off the device. This is a dangerous action and requires user confirmation."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            # The confirmation logic will be in the agent, not the tool itself.
            self.adb_client.poweroff()
            return "Device is powering off."
        except Exception as e:
            logging.error(f"Error in PoweroffTool: {e}")
            return "Failed to power off the device."