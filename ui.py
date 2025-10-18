import os
import json
import time
import logging
from threading import Thread
from queue import Queue, Empty
import streamlit as st
from PIL import Image

from adb_mcp_client import AdbMcpClient
from phone_agent import PhoneAgent

# --- Thread-Safe Queue for UI Updates ---
# The queue will hold tuples of (update_type, data)
# e.g., ("log", "This is a log message") or ("screenshot", "/path/to/image.png")
ui_queue = Queue()

# --- State Management ---
def get_session_state():
    """Initializes and returns the session state."""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "latest_screenshot" not in st.session_state:
        st.session_state.latest_screenshot = None
    return st.session_state

# --- Logging ---
class UILogHandler(logging.Handler):
    """Custom logging handler that puts log records into the UI queue."""
    def emit(self, record):
        log_entry = self.format(record)
        ui_queue.put(("log", log_entry))

def setup_logging():
    """Sets up a logger that outputs to the UI queue."""
    root_logger = logging.getLogger()
    if not any(isinstance(h, UILogHandler) for h in root_logger.handlers):
        root_logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        ui_handler = UILogHandler()
        ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(ui_handler)
        # Also log to console for debugging
        root_logger.addHandler(logging.StreamHandler())

# --- Agent Execution ---
def execute_task_thread(task, max_cycles, config):
    """Runs the PhoneAgent task in a background thread."""
    # This function runs in a separate thread and should not touch st.session_state directly.
    # It communicates with the main thread via the ui_queue.
    
    # Clear the queue for the new task
    while not ui_queue.empty():
        try:
            ui_queue.get_nowait()
        except Empty:
            break

    try:
        ui_queue.put(("log", "Initializing Phone Agent..."))
        agent = PhoneAgent(config)

        # Patch the ADB client's screenshot method to update the UI via the queue
        original_capture = agent.adb_client.capture_screenshot
        def capture_and_update_ui():
            path = original_capture()
            ui_queue.put(("screenshot", path))
            return path
        agent.adb_client.capture_screenshot = capture_and_update_ui

        ui_queue.put(("log", f"Starting task: '{task}'"))
        result = agent.execute_task(task, max_cycles=max_cycles)
        
        if result.get('success'):
            ui_queue.put(("log", f"✅ Task completed successfully: {result.get('result')}"))
        else:
            ui_queue.put(("log", f"❌ Task failed or stopped: {result.get('reason', 'No reason specified.')}"))

    except Exception as e:
        # Log the full exception to the queue
        import traceback
        error_str = f"An error occurred during task execution: {e}\n{traceback.format_exc()}"
        ui_queue.put(("log", error_str))
    finally:
        ui_queue.put(("status", "stopped"))


# --- UI Components ---
def render_sidebar():
    """Renders the Streamlit sidebar for configuration."""
    st.sidebar.title("⚙️ Configuration")

    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=config.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")),
        type="password",
        key="api_key_input"
    )
    model_name = st.sidebar.text_input(
        "Groq Model Name",
        value=config.get("groq_model_name", "llama3-70b-8192"),
        key="model_name_input"
    )
    adb_server = st.sidebar.text_input(
        "ADB MCP Server Address",
        value=config.get("adb_server_address", "http://localhost:8080"),
        key="adb_server_input"
    )

    if st.sidebar.button("Check Connection", use_container_width=True):
        try:
            client = AdbMcpClient(server_address=adb_server)
            device_config = client.get_device_config()
            st.sidebar.success("Connection Successful!")
            st.sidebar.json(device_config)
        except Exception as e:
            st.sidebar.error(f"Connection Failed: {e}")

    max_cycles = st.sidebar.number_input("Max Cycles per Task", min_value=1, max_value=50, value=15, key="max_cycles_input")

    return {
        "groq_api_key": api_key,
        "groq_model_name": model_name,
        "adb_server_address": adb_server
    }, max_cycles

def render_main_content(config, max_cycles):
    """Renders the main content area of the UI."""
    state = get_session_state()
    st.title("📱 Phone Driver UI")
    st.markdown("Control your Android device using natural language, powered by Groq and LangChain.")

    task = st.text_area("Enter your task here:", height=100, placeholder="e.g., 'Open the calculator and add 5 and 7'", key="task_input")

    col1, col2, _ = st.columns([1, 1, 3])
    start_button = col1.button("▶️ Start Task", use_container_width=True, disabled=state.is_running)
    stop_button = col2.button("⏹️ Stop Task", use_container_width=True, disabled=not state.is_running)

    if start_button:
        if not task:
            st.error("Please enter a task.")
        elif not config["groq_api_key"]:
            st.error("Groq API Key is not set. Please add it in the sidebar.")
        else:
            state.is_running = True
            state.logs.clear()
            thread = Thread(target=execute_task_thread, args=(task, max_cycles, config), daemon=True)
            thread.start()
            st.rerun()

    if stop_button:
        state.is_running = False
        # The agent loop will naturally terminate, this just stops the UI refresh
        ui_queue.put(("log", "⏹️ Stop request sent by user."))
        st.rerun()

    st.header("Live View")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Screenshot")
        screenshot_placeholder = st.empty()
        if state.latest_screenshot and os.path.exists(state.latest_screenshot):
            image = Image.open(state.latest_screenshot)
            screenshot_placeholder.image(image, use_column_width=True)
        else:
            screenshot_placeholder.info("Screenshot will appear here when the task runs.")

    with col2:
        st.subheader("Execution Log")
        log_placeholder = st.empty()
        log_placeholder.text_area(
            "Execution Log",  # This is the label
            value="\n".join(state.logs),
            height=400,
            key="log_area",
            label_visibility="collapsed"
        )

def process_ui_queue():
    """Process all pending updates from the UI queue."""
    state = get_session_state()
    while not ui_queue.empty():
        try:
            update_type, data = ui_queue.get_nowait()
            if update_type == "log":
                state.logs.append(data)
                if len(state.logs) > 200:
                    state.logs = state.logs[-200:]
            elif update_type == "screenshot":
                state.latest_screenshot = data
            elif update_type == "status" and data == "stopped":
                state.is_running = False
        except Empty:
            break

# --- Main App ---
def main():
    st.set_page_config(page_title="Phone Driver", layout="wide")
    get_session_state()
    setup_logging()

    config, max_cycles = render_sidebar()

    # Process any updates from the background thread
    process_ui_queue()

    render_main_content(config, max_cycles)

    # Auto-refresh loop only if a task is running
    if get_session_state().is_running:
        time.sleep(1) # Short sleep to avoid busy-waiting
        st.rerun()

if __name__ == "__main__":
    main()