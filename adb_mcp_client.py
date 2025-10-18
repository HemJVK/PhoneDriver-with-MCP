import requests
import logging
from typing import Optional, Dict, Any

class AdbMcpClient:
    """
    Client for interacting with an ADB MCP (Multi-Control Platform) server.
    """
    def __init__(self, server_address: str = "http://localhost:8080"):
        """
        Initializes the client with the server address.

        Args:
            server_address: The base URL of the ADB MCP server.
        """
        self.server_address = server_address
        logging.info(f"ADB MCP Client configured for server at {self.server_address}")

    def _send_command(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a command to the ADB MCP server.

        Args:
            endpoint: The API endpoint to hit (e.g., '/tap').
            payload: The JSON payload for the request.

        Returns:
            The JSON response from the server.
        """
        url = f"{self.server_address}{endpoint}"
        try:
            if payload:
                response = requests.post(url, json=payload)
            else:
                response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send command to {url}: {e}")
            raise

    def capture_screenshot(self) -> str:
        """
        Requests a screenshot from the server.

        Returns:
            The path to the captured screenshot.
        """
        response = self._send_command("/screenshot")
        if not response.get("success") or "path" not in response:
            raise Exception("Failed to capture screenshot from server.")
        return response["path"]

    def tap(self, x: int, y: int):
        """
        Sends a tap command to the server.

        Args:
            x: The x-coordinate.
            y: The y-coordinate.
        """
        self._send_command("/tap", {"x": x, "y": y})

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int):
        """
        Sends a swipe command to the server.

        Args:
            start_x: The starting x-coordinate.
            start_y: The starting y-coordinate.
            end_x: The ending x-coordinate.
            end_y: The ending y-coordinate.
            duration: The duration of the swipe in milliseconds.
        """
        self._send_command("/swipe", {
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "duration": duration
        })

    def type_text(self, text: str):
        """
        Sends a type command to the server.

        Args:
            text: The text to type.
        """
        self._send_command("/type", {"text": text})

    # --- Generic Messaging Methods ---

    def send_message(self, recipient: str, message: str):
        """
        Sends a message to a recipient (user or group) in the current app.

        Args:
            recipient: The name of the user or group to send the message to.
            message: The message content.
        """
        logging.info(f"Sending message to '{recipient}'...")
        self._send_command("/messaging/send", {"recipient": recipient, "message": message})

    def create_group(self, group_name: str, members: list[str]):
        """
        Creates a new group in the current messaging app.

        Args:
            group_name: The name for the new group.
            members: A list of contact names to add to the group initially.
        """
        logging.info(f"Creating group '{group_name}' with members: {members}")
        self._send_command("/messaging/create_group", {"group_name": group_name, "members": members})

    def add_members_to_group(self, group_name: str, members: list[str]):
        """
        Adds members to an existing group in the current messaging app.

        Args:
            group_name: The name of the group to add members to.
            members: A list of contact names to add.
        """
        logging.info(f"Adding members {members} to group '{group_name}'")
        self._send_command("/messaging/add_members", {"group_name": group_name, "members": members})

    # --- System Control Methods ---

    def open_notifications(self):
        self._send_command("/system/notifications")

    def open_quick_settings(self):
        self._send_command("/system/quick_settings")

    def sleep(self):
        self._send_command("/system/sleep")

    def reboot(self):
        self._send_command("/system/reboot")

    def poweroff(self):
        self._send_command("/system/poweroff")

    # --- Communication Methods ---

    def make_call(self, phone_number: str):
        self._send_command("/communication/call", {"phone_number": phone_number})

    def send_sms(self, phone_number: str, message: str):
        self._send_command("/communication/sms", {"phone_number": phone_number, "message": message})

    # --- Camera Methods ---

    def take_photo(self):
        self._send_command("/camera/photo")

    def take_selfie(self):
        self._send_command("/camera/selfie")

    def record_video(self, duration_seconds: int):
        self._send_command("/camera/video", {"duration": duration_seconds})

    def get_device_config(self) -> Dict[str, Any]:
        """
        Retrieves the connected device's configuration from the server.

        Returns:
            A dictionary containing the device's configuration details.
        """
        logging.info("Fetching device configuration from server...")
        return self._send_command("/device/config")