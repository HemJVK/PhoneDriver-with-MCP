import logging
from typing import Type, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from adb_mcp_client import AdbMcpClient

class SendMessageInput(BaseModel):
    recipient: str = Field(description="The name of the user or group to send the message to.")
    message: str = Field(description="The content of the message to send.")

class SendMessageTool(BaseTool):
    name: str = "send_message"
    description: str = "Sends a message to a specified user or group in the current messaging app."
    args_schema: Type[BaseModel] = SendMessageInput
    adb_client: AdbMcpClient

    def _run(self, recipient: str, message: str) -> str:
        try:
            self.adb_client.send_message(recipient, message)
            return f"Successfully sent message to '{recipient}'."
        except Exception as e:
            logging.error(f"Error in SendMessageTool: {e}")
            return f"Failed to send message to '{recipient}'."

class CreateGroupInput(BaseModel):
    group_name: str = Field(description="The name for the new group.")
    members: List[str] = Field(description="A list of contact names to add to the group.")

class CreateGroupTool(BaseTool):
    name: str = "create_group"
    description: str = "Creates a new group with a given name and initial members in the current messaging app."
    args_schema: Type[BaseModel] = CreateGroupInput
    adb_client: AdbMcpClient

    def _run(self, group_name: str, members: List[str]) -> str:
        try:
            self.adb_client.create_group(group_name, members)
            return f"Successfully initiated creation of group '{group_name}'."
        except Exception as e:
            logging.error(f"Error in CreateGroupTool: {e}")
            return f"Failed to create group '{group_name}'."

class AddMembersToGroupInput(BaseModel):
    group_name: str = Field(description="The name of the existing group to add members to.")
    members: List[str] = Field(description="A list of contact names to add.")

class AddMembersToGroupTool(BaseTool):
    name: str = "add_members_to_group"
    description: str = "Adds new members to an existing group in the current messaging app."
    args_schema: Type[BaseModel] = AddMembersToGroupInput
    adb_client: AdbMcpClient

    def _run(self, group_name: str, members: List[str]) -> str:
        try:
            self.adb_client.add_members_to_group(group_name, members)
            return f"Successfully added members to group '{group_name}'."
        except Exception as e:
            logging.error(f"Error in AddMembersToGroupTool: {e}")
            return f"Failed to add members to group '{group_name}'."