import logging
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class MakeCallInput(BaseModel):
    phone_number: str = Field(description="The phone number to call.")

class MakeCallTool(BaseTool):
    name: str = "make_phone_call"
    description: str = "Initiates a phone call to the specified number using the default dialer."
    args_schema: Type[BaseModel] = MakeCallInput
    adb_client: AdbMcpClient

    def _run(self, phone_number: str) -> str:
        try:
            self.adb_client.make_call(phone_number)
            return f"Successfully initiated call to {phone_number}."
        except Exception as e:
            logging.error(f"Error in MakeCallTool: {e}")
            return f"Failed to initiate call to {phone_number}."

class SendSmsInput(BaseModel):
    phone_number: str = Field(description="The recipient's phone number.")
    message: str = Field(description="The content of the SMS message.")

class SendSmsTool(BaseTool):
    name: str = "send_sms_message"
    description: str = "Sends an SMS message to the specified number using the default messaging app."
    args_schema: Type[BaseModel] = SendSmsInput
    adb_client: AdbMcpClient

    def _run(self, phone_number: str, message: str) -> str:
        try:
            self.adb_client.send_sms(phone_number, message)
            return f"Successfully sent SMS to {phone_number}."
        except Exception as e:
            logging.error(f"Error in SendSmsTool: {e}")
            return f"Failed to send SMS to {phone_number}."