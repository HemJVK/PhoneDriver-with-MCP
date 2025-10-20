import json
import logging
import streamlit as st
import subprocess
import os
from phone_agent import PhoneAgent
from adb_controller import AdbController

# --- Logging Setup ---
class UILogHandler(logging.Handler):
    """A logging handler that writes logs to a Streamlit UI element."""
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.buffer = []

    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        self.container.code("\n".join(self.buffer))

def setup_logging(container):
    """Sets up a logger to stream to the UI."""
    root_logger = logging.getLogger()
    # Clear any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_handler = UILogHandler(container)
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)

# --- Config Management ---
def load_config():
    """Loads config.json or returns a default."""
    default_config = {
        "device_id": None,
        # Add LLM temp to config
        "temperature": 0.1
    }
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            # Ensure all keys from default are present
            for key, value in default_config.items():
                config.setdefault(key, value)
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config

def save_config(config):
    """Saves the config to config.json."""
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("📱 Phone Agent Control")

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Create tabs
query_tab, config_tab = st.tabs(["Query", "Configuration"])

# --- Configuration Tab ---
with config_tab:
    st.header("⚙️ Configuration")
    
    st.subheader("Device Detection")
    if st.button("Detect Phone"):
        try:
            controller = AdbController()
            st.session_state.config['device_id'] = controller.device_id
            st.success(f"Phone detected with screen resolution: {controller.width}x{controller.height}")
            save_config(st.session_state.config)
        except Exception as e:
            st.error(f"Could not detect phone. Please ensure ADB is installed and your device is connected. Error: {e}")

    st.subheader("LLM Configuration")
    temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config.get("temperature", 0.1),
        step=0.05,
        help="Controls the randomness of the LLM's responses. Lower is more deterministic."
    )
    st.session_state.config["temperature"] = temp

    if st.button("Save Configuration"):
        save_config(st.session_state.config)
        st.success("Configuration saved successfully!")
        # Clear the agent to force re-initialization with new settings
        st.session_state.agent = None

# --- Query Tab ---
with query_tab:
    st.header("▶️ Run a Task")

    query = st.text_area("Enter your query for the phone agent:", height=100)

    run_button = st.button("Execute Task")

    st.subheader("Agent Logs")
    log_container = st.empty()
    log_box = log_container.code("Logs will appear here...", language="log")

    setup_logging(log_container=log_container)

    if run_button and query:
        with st.spinner("Agent is running... Please wait."):
            try:
                # Initialize agent if it doesn't exist
                if st.session_state.agent is None:
                    logging.info("Initializing agent...")
                    st.session_state.agent = PhoneAgent(config=st.session_state.config)

                logging.info(f"Executing task: {query}")
                result = st.session_state.agent.execute_task(query)
                logging.info(f"Task finished with result: {result}")
                st.success("Task execution finished!")

            except Exception as e:
                logging.error(f"Failed to execute task: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")
    elif run_button:
        st.warning("Please enter a query.")
