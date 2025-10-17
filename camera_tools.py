import logging
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class TakePhotoTool(BaseTool):
    name: str = "take_photo"
    description: str = "Opens the camera and takes a photo using the rear-facing camera."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            self.adb_client.take_photo()
            return "Successfully took a photo."
        except Exception as e:
            logging.error(f"Error in TakePhotoTool: {e}")
            return "Failed to take a photo."

class TakeSelfieTool(BaseTool):
    name: str = "take_selfie"
    description: str = "Opens the camera and takes a selfie using the front-facing camera."
    adb_client: AdbMcpClient

    def _run(self) -> str:
        try:
            self.adb_client.take_selfie()
            return "Successfully took a selfie."
        except Exception as e:
            logging.error(f"Error in TakeSelfieTool: {e}")
            return "Failed to take a selfie."

class RecordVideoInput(BaseModel):
    duration_seconds: int = Field(description="The duration of the video recording in seconds.")

class RecordVideoTool(BaseTool):
    name: str = "record_video"
    description: str = "Opens the camera and records a video for a specified duration."
    args_schema: Type[BaseModel] = RecordVideoInput
    adb_client: AdbMcpClient

    def _run(self, duration_seconds: int) -> str:
        try:
            self.adb_client.record_video(duration_seconds)
            return f"Successfully recorded a {duration_seconds}-second video."
        except Exception as e:
            logging.error(f"Error in RecordVideoTool: {e}")
            return "Failed to record video."