import logging
from typing import Type, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class SendWhatsAppMessageInput(BaseModel):
    recipient: str = Field(description="The name of the user or group to send the message to.")
    message: str = Field(description="The content of the message to send.")

class SendWhatsAppMessageTool(BaseTool):
    name = "send_whatsapp_message"
    description = "Sends a message to a specified user or group in WhatsApp."
    args_schema: Type[BaseModel] = SendWhatsAppMessageInput
    adb_client: AdbMcpClient

    def _run(self, recipient: str, message: str) -> str:
        try:
            self.adb_client.send_whatsapp_message(recipient, message)
            return f"Successfully sent message to '{recipient}'."
        except Exception as e:
            logging.error(f"Error in SendWhatsAppMessageTool: {e}")
            return f"Failed to send message to '{recipient}'."

class CreateWhatsAppGroupInput(BaseModel):
    group_name: str = Field(description="The name for the new WhatsApp group.")
    members: List[str] = Field(description="A list of contact names to add to the group.")

class CreateWhatsAppGroupTool(BaseTool):
    name = "create_whatsapp_group"
    description = "Creates a new WhatsApp group with a given name and initial members."
    args_schema: Type[BaseModel] = CreateWhatsAppGroupInput
    adb_client: AdbMcpClient

    def _run(self, group_name: str, members: List[str]) -> str:
        try:
            self.adb_client.create_whatsapp_group(group_name, members)
            return f"Successfully initiated creation of group '{group_name}'."
        except Exception as e:
            logging.error(f"Error in CreateWhatsAppGroupTool: {e}")
            return f"Failed to create group '{group_name}'."

class AddMembersToGroupInput(BaseModel):
    group_name: str = Field(description="The name of the existing group to add members to.")
    members: List[str] = Field(description="A list of contact names to add.")

class AddMembersToGroupTool(BaseTool):
    name = "add_members_to_whatsapp_group"
    description = "Adds new members to an existing WhatsApp group."
    args_schema: Type[BaseModel] = AddMembersToGroupInput
    adb_client: AdbMcpClient

    def _run(self, group_name: str, members: List[str]) -> str:
        try:
            self.adb_client.add_members_to_whatsapp_group(group_name, members)
            return f"Successfully added members to group '{group_name}'."
        except Exception as e:
            logging.error(f"Error in AddMembersToGroupTool: {e}")
            return f"Failed to add members to group '{group_name}'."