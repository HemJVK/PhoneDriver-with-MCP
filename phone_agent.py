import os
import json
import logging
from groq_agent import GroqAgent

def main():
    """
    Main function to run the phone agent.
    """
    if len(os.sys.argv) < 2:
        print("Usage: python phone_agent.py 'your task here'")
        os.sys.exit(1)
    
    task = ' '.join(os.sys.argv[1:])
    
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
        
    device_id = config.get('device_id')

    agent = GroqAgent(api_key=groq_api_key, device_id=device_id)
    
    result = agent.run_task(task)
    
    print("\n----- Task Complete -----")
    print(f"Final Result: {result}")
    print("-------------------------")

if __name__ == "__main__":
    main()
