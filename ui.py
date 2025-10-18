import os
import json
import time
import logging
from threading import Thread
import streamlit as st
from PIL import Image

from adb_mcp_client import AdbMcpClient
from phone_agent import PhoneAgent

# --- State Management ---
def get_session_state():
    """Initializes and returns the session state."""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "latest_screenshot" not in st.session_state:
        st.session_state.latest_screenshot = None
    return st.session_state

# --- Logging ---
class UILogHandler(logging.Handler):
    """Custom logging handler that writes logs to the Streamlit UI."""
    def emit(self, record):
        log_entry = self.format(record)
        st.session_state.logs.append(log_entry)
        # Keep logs list from growing indefinitely
        if len(st.session_state.logs) > 200:
            st.session_state.logs = st.session_state.logs[-200:]

def setup_logging():
    """Sets up a logger that outputs to the UI."""
    root_logger = logging.getLogger()
    if not any(isinstance(h, UILogHandler) for h in root_logger.handlers):
        root_logger.setLevel(logging.INFO)
        ui_handler = UILogHandler()
        ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(ui_handler)

# --- Agent Execution ---
def execute_task_thread(task, max_cycles, config):
    """Runs the PhoneAgent task in a background thread."""
    state = get_session_state()
    state.is_running = True
    state.logs.clear()
    
    try:
        logging.info("Initializing Phone Agent...")
        agent = PhoneAgent(config)
        st.session_state.agent = agent

        # Patch the ADB client's screenshot method to update the UI
        original_capture = agent.adb_client.capture_screenshot
        def capture_and_update_ui():
            path = original_capture()
            st.session_state.latest_screenshot = path
            return path
        agent.adb_client.capture_screenshot = capture_and_update_ui

        logging.info(f"Starting task: '{task}'")
        result = agent.execute_task(task, max_cycles=max_cycles)
        
        if result.get('success'):
            logging.info(f"Task completed successfully: {result.get('result')}")
        else:
            logging.error(f"Task failed or stopped: {result.get('reason', 'No reason specified.')}")

    except Exception as e:
        logging.error(f"An error occurred during task execution: {e}", exc_info=True)
    finally:
        state.is_running = False

# --- UI Components ---
def render_sidebar():
    """Renders the Streamlit sidebar for configuration."""
    st.sidebar.title("⚙️ Configuration")

    # Load existing config or use defaults
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Get API key from env var as default
    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=config.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")),
        type="password"
    )
    model_name = st.sidebar.text_input(
        "Groq Model Name",
        value=config.get("groq_model_name", "llama3-70b-8192")
    )
    adb_server = st.sidebar.text_input(
        "ADB MCP Server Address",
        value=config.get("adb_server_address", "http://localhost:8080")
    )

    if st.sidebar.button("Check Connection", use_container_width=True):
        try:
            client = AdbMcpClient(server_address=adb_server)
            device_config = client.get_device_config()
            st.sidebar.success("Connection Successful!")
            st.sidebar.json(device_config)
        except Exception as e:
            st.sidebar.error(f"Connection Failed: {e}")

    max_cycles = st.sidebar.number_input("Max Cycles per Task", min_value=1, max_value=50, value=15)

    return {
        "groq_api_key": api_key,
        "groq_model_name": model_name,
        "adb_server_address": adb_server
    }, max_cycles

def render_main_content(config, max_cycles):
    """Renders the main content area of the UI."""
    st.title("📱 Phone Driver UI")
    st.markdown("Control your Android device using natural language, powered by Groq and LangChain.")

    task = st.text_area("Enter your task here:", height=100, placeholder="e.g., 'Open the calculator and add 5 and 7'")

    col1, col2, _ = st.columns([1, 1, 3])
    start_button = col1.button("▶️ Start Task", use_container_width=True)
    stop_button = col2.button("⏹️ Stop Task", use_container_width=True)

    if start_button and not st.session_state.is_running:
        if not task:
            st.error("Please enter a task.")
        elif not config["groq_api_key"]:
            st.error("Groq API Key is not set. Please add it in the sidebar.")
        else:
            # Start the agent in a new thread
            thread = Thread(target=execute_task_thread, args=(task, max_cycles, config))
            thread.daemon = True
            thread.start()
            st.success("Task started! See logs and screenshots below.")

    if stop_button and st.session_state.is_running:
        st.session_state.is_running = False
        # A bit of a hack to stop the agent; the loop in execute_task should check this flag
        if st.session_state.agent:
             # This assumes the agent's loop will naturally terminate.
             # For a more forceful stop, a more complex mechanism would be needed.
            logging.warning("Stop requested. The current cycle will finish.")
        st.warning("Task stop requested. It will halt after the current step.")

    st.header("Live View")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Screenshot")
        screenshot_placeholder = st.empty()
        if st.session_state.latest_screenshot and os.path.exists(st.session_state.latest_screenshot):
            image = Image.open(st.session_state.latest_screenshot)
            screenshot_placeholder.image(image, use_column_width=True)
        else:
            screenshot_placeholder.info("Screenshot will appear here when the task runs.")

    with col2:
        st.subheader("Execution Log")
        log_placeholder = st.empty()
        log_placeholder.text_area("", value="\n".join(st.session_state.logs), height=400, key="log_area")

# --- Main App ---
def main():
    st.set_page_config(page_title="Phone Driver", layout="wide")
    get_session_state()
    setup_logging()

    config, max_cycles = render_sidebar()
    render_main_content(config, max_cycles)

    # Auto-refresh loop
    if st.session_state.is_running:
        time.sleep(2)
        st.experimental_rerun()

if __name__ == "__main__":
    main()