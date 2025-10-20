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
    default_config = { "device_id": None, "text_model": "gemma-7b-it" }
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

# Initialize session state for multi-step workflows
if 'config' not in st.session_state: st.session_state.config = load_config()
if 'agent' not in st.session_state: st.session_state.agent = None
if 'plan' not in st.session_state: st.session_state.plan = None
if 'run_approved' not in st.session_state: st.session_state.run_approved = False

query_tab, config_tab = st.tabs(["Query", "Configuration"])

with config_tab:
    st.header("⚙️ Configuration")
    st.subheader("LLM Configuration")
    text_models = ["gemma-7b-it"]
    text_model_name = st.selectbox("Select Text (Reasoning) Model", options=text_models, index=0)
    st.session_state.config["text_model"] = text_model_name
    if st.button("Save Configuration"):
        save_config(st.session_state.config)
        st.success("Configuration saved!")
        st.session_state.agent = None

with query_tab:
    st.header("▶️ Run a Task")

    query = st.text_area("Enter your query:", height=100)
    review_mode = st.toggle("Review Steps Before Execution", value=True)

    # Logic for starting the task (either generating a plan or running directly)
    if st.button("Execute Task") and query:
        log_handler.buffer.clear()
        logging.info("Initializing agent...")
        st.session_state.agent = PhoneAgent(config=st.session_state.config)

        if review_mode:
            with st.spinner("Generating plan..."):
                st.session_state.plan = st.session_state.agent.agent.generate_plan(query)
        else:
            # Run autonomously
            with st.spinner("Agent is running autonomously..."):
                plan = st.session_state.agent.agent.generate_plan(query)
                result = st.session_state.agent.agent.run_task_loop(query, plan=plan)
                logging.info(f"Task finished with result: {result}")
                st.success("Task execution finished!")

    # Logic for the review form
    if st.session_state.plan:
        with st.form("review_plan_form"):
            st.subheader("Review and Edit the Plan")
            edited_plan = st.text_area("Generated Plan:", value=st.session_state.plan, height=200)

            if st.form_submit_button("Approve and Run"):
                st.session_state.plan = edited_plan # Save edits
                st.session_state.run_approved = True
                st.session_state.plan = None # Clear the plan to hide the form

    # Logic for executing after approval
    if st.session_state.run_approved:
        with st.spinner("Executing approved plan..."):
            logging.info("User approved the plan. Executing...")
            result = st.session_state.agent.agent.run_task_loop(query, plan=st.session_state.plan)
            logging.info(f"Task finished with result: {result}")
            st.success("Task execution finished!")
        st.session_state.run_approved = False # Reset the flag

    st.subheader("Agent Logs")
    log_container = st.empty()
    log_handler = setup_logging(log_container)

