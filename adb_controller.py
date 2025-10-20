import subprocess
import logging
import re
import os
import time

class AdbController:
    """A wrapper for ADB commands to control an Android device."""
    def __init__(self, device_id: str = None):
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        self.width, self.height = self._get_screen_resolution()

    def _get_screen_resolution(self) -> tuple[int, int]:
        """Gets the device's screen resolution using ADB."""
        try:
            output = self.run_adb_command("shell wm size")
            match = re.search(r'Physical size: (\d+)x(\d+)', output)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                self.logger.info(f"Detected screen resolution: {width}x{height}")
                return width, height
            raise RuntimeError("Could not parse screen resolution from ADB output.")
        except Exception as e:
            self.logger.warning(f"Could not get screen resolution, defaulting to 1080x2340. Error: {e}")
            return 1080, 2340

    def run_adb_command(self, command: str) -> str:
        """Executes an ADB command and returns the output."""
        device_prefix = f"-s {self.device_id}" if self.device_id else ""
        full_command = f"adb {device_prefix} {command}"
        try:
            result = subprocess.run(
                full_command, shell=True, check=True, capture_output=True, text=True
            )
            self.logger.info(f"ADB command successful: {full_command}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ADB command failed: {full_command}\nError: {e.stderr}")
            raise

    def tap(self, x: int, y: int):
        self.run_adb_command(f"shell input tap {x} {y}")

    def swipe(self, direction: str):
        center_x, center_y = self.width // 2, self.height // 2
        swipe_distance_y = int(self.height * 0.4)
        swipe_distance_x = int(self.width * 0.4)

        if direction == "up":
            self.run_adb_command(f"shell input swipe {center_x} {center_y + swipe_distance_y} {center_x} {center_y - swipe_distance_y} 300")
        elif direction == "down":
            self.run_adb_command(f"shell input swipe {center_x} {center_y - swipe_distance_y} {center_x} {center_y + swipe_distance_y} 300")
        elif direction == "left":
            self.run_adb_command(f"shell input swipe {center_x + swipe_distance_x} {center_y} {center_x - swipe_distance_x} {center_y} 300")
        elif direction == "right":
            self.run_adb_command(f"shell input swipe {center_x - swipe_distance_x} {center_y} {center_x + swipe_distance_x} {center_y} 300")
        else:
            raise ValueError(f"Invalid swipe direction: {direction}")

    def type_text(self, text: str):
        escaped_text = text.replace(" ", "%s").replace("'", "\\'")
        self.run_adb_command(f'shell input text "{escaped_text}"')

    def open_notifications(self):
        self.run_adb_command("shell cmd statusbar expand-notifications")

    def make_call(self, number: str):
        self.run_adb_command(f"shell am start -a android.intent.action.CALL -d tel:{number}")

    def send_message(self, number: str, message: str):
        self.run_adb_command(f"shell am start -a android.intent.action.SENDTO -d sms:{number}")
        self.run_adb_command(f"shell input text '{message.replace(' ', '%s')}'")
        self.run_adb_command("shell input keyevent 22")
        self.run_adb_command("shell input keyevent 66")

    def system_command(self, command: str):
        logging.warning(f"Executing potentially disruptive system command: '{command}'.")
        if command == "sleep":
            self.run_adb_command("shell input keyevent 26")
        elif command == "reboot":
            self.run_adb_command("reboot")
        elif command == "poweroff":
            self.run_adb_command("reboot -p")
        else:
            raise ValueError(f"Unsupported system command: {command}")

    def take_photo(self):
        self.run_adb_command("shell am start -a android.media.action.IMAGE_CAPTURE")
        self.run_adb_command("shell input keyevent 27")
