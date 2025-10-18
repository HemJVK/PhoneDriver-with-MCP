# Phone Driver

A Python-based mobile automation agent that uses a Groq-powered LangChain agent to understand and interact with Android devices through visual analysis.

<p align="center">
  <img src="Images/PhoneDriver.png" width="600" alt="Phone Driver Demo">
</p>

## Features

- 🤖 **Cloud-Powered Vision**: Uses Groq's Llama 3 for fast and powerful visual analysis.
- 🔗 **LangChain Integration**: Employs a robust, tool-based architecture for reliable action execution.
- 📱 **Client-Server ADB**: Interacts with Android devices via a separate ADB MCP server for improved stability.
- 🎯 **Natural Language Tasks**: Describe what you want in plain English.
- 🖥️ **Web UI**: A simple Gradio interface is available for easy control.
- 📊 **Real-time Feedback**: Provides live execution logs.

## Architecture

This application has been refactored to a modern, modular architecture:

1.  **PhoneAgent (`phone_agent.py`)**: The main orchestrator that manages the task lifecycle.
2.  **GroqAgent (`groq_agent.py`)**: A LangChain-based agent that analyzes screenshots and user requests to decide the next action. It uses a vision-capable model hosted on Groq.
3.  **LangChain Tools (`langchain_tools.py`)**: A set of custom tools (`TapTool`, `SwipeTool`, etc.) that the agent can use to interact with the device.
4.  **ADB MCP Client (`adb_mcp_client.py`)**: A client that sends commands (e.g., tap, swipe) to a separate ADB MCP server. **This server is a required component that you must run separately.** It is responsible for translating these commands into actual ADB commands.

## Requirements

- Python 3.10+
- An active Groq API key.
- A running ADB MCP server.
- Android device with USB debugging & Developer Mode enabled.
- ADB (Android Debug Bridge) installed on the machine running the MCP server.

## Installation

### 1. Clone Repo & Install Python Dependencies

```bash
git clone https://github.com/OminousIndustries/PhoneDriver.git
cd PhoneDriver
```
Create and activate a virtual environment:

```bash
python -m venv phonedriver
source phonedriver/bin/activate
```
Install Python dependencies:

```bash
pip install -r requirements.txt
# You will need to create a requirements.txt file with:
# langchain
# langchain-groq
# pydantic
# requests
# pillow
# gradio
```

### 2. Set Up Groq API Key

You must have a Groq API key to use the agent.

1.  Obtain an API key from the [Groq website](https://console.groq.com/keys).
2.  Set it as an environment variable:
    ```bash
    export GROQ_API_KEY="your-groq-api-key"
    ```
    Alternatively, you can add it to a `config.json` file.

### 3. Set Up and Run the ADB MCP Server

This application requires a separate server to handle ADB commands. You are responsible for creating or obtaining this server. The server must expose the following endpoints:

-   `GET /screenshot`: Captures a screenshot and returns a JSON with `{"success": true, "path": "/path/to/screenshot.png"}`.
-   `POST /tap`: Accepts `{"x": int, "y": int}` to perform a tap.
-   `POST /swipe`: Accepts `{"start_x": int, "start_y": int, "end_x": int, "end_y": int, "duration": int}`.
-   `POST /type`: Accepts `{"text": "string"}` to type text.

#### Generic Messaging Endpoints (New)
-   `POST /messaging/send`: Accepts `{"recipient": "string", "message": "string"}`.
-   `POST /messaging/create_group`: Accepts `{"group_name": "string", "members": ["list", "of", "strings"]}`.
-   `POST /messaging/add_members`: Accepts `{"group_name": "string", "members": ["list", "of", "strings"]}`.

#### System Control Endpoints (New)
-   `GET /system/notifications`: Opens the notification shade.
-   `GET /system/quick_settings`: Opens the quick settings panel.
-   `GET /system/sleep`: Puts the device to sleep.
-   `POST /system/reboot`: Reboots the device.
-   `POST /system/poweroff`: Powers off the device.

#### Communication Endpoints (New)
-   `POST /communication/call`: Accepts `{"phone_number": "string"}`.
-   `POST /communication/sms`: Accepts `{"phone_number": "string", "message": "string"}`.

#### Camera Endpoints (New)
-   `POST /camera/photo`: Takes a photo with the rear camera.
-   `POST /camera/selfie`: Takes a photo with the front camera.
-   `POST /camera/video`: Accepts `{"duration": int}` (in seconds).

#### Device Endpoints (New)
-   `GET /device/config`: Retrieves the configuration of the connected device. Expected successful response: `{"device_id": "...", "screen_width": 1080, "screen_height": 2340}`.

Start your ADB MCP server and ensure it is accessible from where you are running the PhoneDriver.

## Configuration

You can configure the application using a `config.json` file in the root directory.

```json
{
  "adb_server_address": "http://localhost:8080",
  "groq_model_name": "llama3-70b-8192",
  "groq_api_key": "your-groq-api-key-if-not-using-env-var"
}
```

## Usage

### Web UI (Recommended)

Launch the Gradio interface:

```bash
python ui.py
```
Open your browser to `http://localhost:7860` and enter your task.

### Command Line

```bash
python phone_agent.py "your task here"
```
Example:
```bash
python phone_agent.py "Open the calculator and add 5 and 7"
```

## How It Works

1.  **Screenshot**: The `PhoneAgent` requests a screenshot from the ADB MCP server.
2.  **Analysis**: The screenshot, user request, and action history are sent to the `GroqAgent`.
3.  **Decision**: The Groq LLM decides which tool to use (e.g., `tap`, `type`) and with what arguments.
4.  **Execution**: The `PhoneAgent` runs the selected tool, which uses the `AdbMcpClient` to send the command to the ADB server.
5.  **Repeat**: The cycle continues until the agent decides the task is complete by using the `terminate` tool.

## License

Apache License 2.0 - see LICENSE file for details.