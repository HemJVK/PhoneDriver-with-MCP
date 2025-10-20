import json
import logging
import streamlit as st
from phone_agent import PhoneAgent

# --- Logging Setup ---
class UILogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.buffer = []

    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        if len(self.buffer) > 200: self.buffer = self.buffer[-200:]
        self.container.code("\n".join(self.buffer), language="log")

def setup_logging(container):
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()

    log_handler = UILogHandler(container)
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)
    return log_handler

# --- Config Management ---
def load_config():
    default_config = {
        "device_id": None,
        "text_model": "gemma-7b-it", # A valid, working default
    }
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            for key, value in default_config.items():
                config.setdefault(key, value)
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("📱 Phone Agent Control")

if 'config' not in st.session_state: st.session_state.config = load_config()
if 'agent' not in st.session_state: st.session_state.agent = None

query_tab, config_tab = st.tabs(["Query", "Configuration"])

with config_tab:
    st.header("⚙️ Configuration")
    
    st.subheader("LLM Configuration")

    # A curated list of known good models
    text_models = ["gemma-7b-it"]

    text_model_name = st.selectbox(
        "Select Text (Reasoning) Model",
        options=text_models,
        index=text_models.index(st.session_state.config.get("text_model", "gemma-7b-it")),
        help="Select a powerful model for the best tool-use performance."
    )
    st.session_state.config["text_model"] = text_model_name

    if st.button("Save Configuration"):
        save_config(st.session_state.config)
        st.success("Configuration saved! Agent will use new settings on next run.")
        st.session_state.agent = None

with query_tab:
    st.header("▶️ Run a Task")

    query = st.text_area("Enter your query:", height=100)

    run_button = st.button("Execute Task")

    st.subheader("Agent Logs")
    log_container = st.empty()
    log_handler = setup_logging(log_container)

    if run_button and query:
        log_handler.buffer.clear()
        with st.spinner("Agent is running..."):
            try:
                logging.info("Initializing agent with current settings...")
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
