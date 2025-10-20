import json
import logging
import threading
import gradio as gr
from phone_agent import PhoneAgent

class UILogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

def setup_logging():
    log_handler = UILogHandler()
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)
    return log_handler

log_handler = setup_logging()
agent_instance = None
is_running = False

def get_agent():
    global agent_instance
    if agent_instance is None:
        logging.info("Initializing agent for the first time...")
        config = {}
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.warning("config.json not found, using default device.")

        agent_instance = PhoneAgent(config=config)
    return agent_instance

def execute_task_thread(task_text):
    global is_running
    is_running = True
    log_handler.logs.clear()
    
    try:
        agent = get_agent()
        logging.info(f"Starting task: {task_text}")
        result = agent.execute_task(task_text)
        logging.info(f"Task finished. Final result: {result}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        is_running = False

def start_task(task_text):
    global is_running
    if is_running:
        gr.Warning("A task is already running.")
        return "", ""
    
    if not task_text:
        gr.Warning("Please enter a task.")
        return "", ""

    thread = threading.Thread(target=execute_task_thread, args=(task_text,))
    thread.start()
    
    return "Task started... Logs will update automatically.", ""

def get_logs():
    return "\n".join(log_handler.logs)

def create_ui():
    with gr.Blocks(title="Phone Agent", theme="soft") as demo:
        gr.Markdown("# 📱 Phone Agent Control")
        gr.Markdown("A text-based agent to control your Android device.")

        with gr.Row():
            with gr.Column(scale=2):
                task_input = gr.Textbox(label="Task to Perform", placeholder="e.g., 'open the calculator and type 2+2'")
                start_button = gr.Button("▶️ Run Task", variant="primary")
            
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="Agent Logs", lines=15, autoscroll=True, interactive=False)
        
        start_button.click(
            fn=start_task,
            inputs=[task_input],
            outputs=[task_input, log_output]
        )
        
        # Correct way to schedule periodic updates in Gradio
        gr.Timer(1, every=None).tick(
            fn=get_logs,
            outputs=log_output,
        )
        
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.queue()
    ui.launch(show_error=True)
