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