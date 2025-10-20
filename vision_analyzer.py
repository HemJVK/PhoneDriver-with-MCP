import logging
from groq import Groq
import base64
from PIL import Image

class VisionAnalyzer:
    """
    A class to analyze screenshots using a Groq vision-language model (VLM).
    Its primary role is to describe the contents of the screen for another agent.
    """
    def __init__(self, api_key: str, model_name: str = "l4-scout-17b"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.client = Groq(api_key=api_key)
        self.prompt = (
            "You are a concise UI description assistant. Your task is to analyze a screenshot of a phone screen "
            "and provide a structured description of all interactive elements (buttons, text fields, icons). "
            "For each element, state its purpose and provide its center (x, y) coordinates. "
            "Be brief and factual. The goal is to give a text-only LLM the necessary information to choose the next action."
        )

    def describe_screen(self, screenshot_path: str) -> str:
        """
        Takes a path to a screenshot and returns a text description of its contents.
        """
        self.logger.info(f"Analyzing screenshot with Groq vision model: {self.model_name}")
        try:
            with open(screenshot_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_base64",
                                "image_base64": encoded_image,
                            },
                            {
                                "type": "text",
                                "text": "Describe the elements on this screen.",
                            },
                        ],
                    },
                ],
                model=self.model_name,
            )

            description = chat_completion.choices[0].message.content
            self.logger.info(f"Screen description generated: {description}")
            return description

        except Exception as e:
            self.logger.error(f"Failed to describe screen with Groq vision model: {e}", exc_info=True)
            return "Error: Could not analyze the screen."
