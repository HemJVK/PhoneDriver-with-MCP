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

    def set_container(self, container):
        self.container = container
        if self.container: self.container.code("\n".join(self.buffer), language="log")

    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        if len(self.buffer) > 200: self.buffer = self.buffer[-200:]
        if self.container: self.container.code("\n".join(self.buffer), language="log")

# Initialize logger globally via session state to prevent re-creation
if 'log_handler' not in st.session_state:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    st.session_state.log_handler = UILogHandler(None)
    st.session_state.log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    st.session_state.log_handler.setFormatter(formatter)
    root_logger.addHandler(st.session_state.log_handler)
    root_logger.setLevel(logging.INFO)

# --- Config Management ---
def load_config():
    default_config = { "device_id": None, "text_model": "gemma-7b-it" }
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            for key, value in default_config.items(): config.setdefault(key, value)
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config

def save_config(config):
    with open('config.json', 'w') as f: json.dump(config, f, indent=2)

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("📱 Phone Agent Control")

# Initialize all necessary session state keys
if 'config' not in st.session_state: st.session_state.config = load_config()
if 'agent' not in st.session_state: st.session_state.agent = None
if 'plan' not in st.session_state: st.session_state.plan = None
if 'run_approved' not in st.session_state: st.session_state.run_approved = False
if 'query' not in st.session_state: st.session_state.query = ""

query_tab, config_tab = st.tabs(["Query", "Configuration"])

with config_tab:
    st.header("⚙️ Configuration")
    st.subheader("LLM Configuration")
    text_models = ["gemma-7b-it"]
    text_model_name = st.selectbox("Select Text Model", options=text_models, index=0)
    st.session_state.config["text_model"] = text_model_name
    if st.button("Save Configuration"):
        save_config(st.session_state.config)
        st.success("Configuration saved!")
        st.session_state.agent = None

with query_tab:
    st.header("▶️ Run a Task")
    
    query_input = st.text_area("Enter your query:", height=100, key="query_text_area")
    review_mode = st.toggle("Review Steps Before Execution", value=True)
    
    if st.button("Execute Task") and query_input:
        st.session_state.query = query_input
        st.session_state.log_handler.buffer.clear()
        logging.info("Initializing agent...")
        st.session_state.agent = PhoneAgent(config=st.session_state.config)
        
        if review_mode:
            with st.spinner("Generating plan..."):
                st.session_state.plan = st.session_state.agent.agent.generate_plan(st.session_state.query)
        else:
            st.session_state.run_approved = True # Skip review

    if st.session_state.plan:
        with st.form("review_plan_form"):
            st.subheader("Review and Edit the Plan")
            edited_plan = st.text_area("Generated Plan:", value=st.session_state.plan, height=200)
            if st.form_submit_button("Approve and Run"):
                st.session_state.plan = edited_plan
                st.session_state.run_approved = True
    
    if st.session_state.run_approved:
        plan_to_run = st.session_state.plan if st.session_state.plan else "Autonomous execution"
        with st.spinner("Executing plan..."):
            logging.info("Executing task...")
            # Ensure agent is initialized
            if st.session_state.agent is None:
                st.session_state.agent = PhoneAgent(config=st.session_state.config)
            result = st.session_state.agent.agent.run_task_loop(st.session_state.query, plan=plan_to_run)
            logging.info(f"Task finished with result: {result}")
            st.success("Task execution finished!")
        # Reset state after run
        st.session_state.run_approved = False
        st.session_state.plan = None

    st.subheader("Agent Logs")
    log_container = st.empty()
    st.session_state.log_handler.set_container(log_container)
