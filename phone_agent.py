import os
import json
import logging
from groq_agent import GroqAgent
from dotenv import load_dotenv

class PhoneAgent:
    """
    A wrapper class for the GroqAgent to provide a consistent interface.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the PhoneAgent.
        """
        if config is None:
            config = {}

        # Load environment variables from .env file
        load_dotenv()

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            # The error is now more informative.
            raise ValueError("GROQ_API_KEY not found. Please create a .env file in the root directory and add GROQ_API_KEY='your_api_key'.")

        device_id = config.get('device_id')

        self.agent = GroqAgent(api_key=groq_api_key, device_id=device_id)
        logging.info("PhoneAgent initialized with GroqAgent.")

    def execute_task(self, user_request: str):
        """
        Executes a task using the underlying GroqAgent.
        """
        return self.agent.run_task(user_request)

def main():
    """
    Main function to run the phone agent from the command line.
    """
    if len(os.sys.argv) < 2:
        print("Usage: python phone_agent.py 'your task here'")
        os.sys.exit(1)
    
    task = ' '.join(os.sys.argv[1:])
    
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
    
    try:
        agent = PhoneAgent(config)
        result = agent.execute_task(task)

        print("\n----- Task Complete -----")
        print(f"Final Result: {result}")
        print("-------------------------")
    except ValueError as e:
        print(f"Error: {e}")
        # Add a hint for the user
        print("Please make sure you have a .env file with your GROQ_API_KEY.")

if __name__ == "__main__":
    main()
