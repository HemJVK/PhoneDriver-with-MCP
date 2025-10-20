import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class VisionAnalyzer:
    """
    A class to analyze screenshots using the Qwen-VL vision-language model.
    Its primary role is to describe the contents of the screen for the reasoning agent.
    """
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing VisionAnalyzer on device: {self.device}")

        # Use trust_remote_code=True for this specific model architecture
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.prompt = (
            "You are a concise UI description assistant. Your task is to analyze a screenshot of a phone screen "
            "and provide a structured description of all interactive elements (buttons, text fields, icons). "
            "For each element, state its purpose and provide its approximate center (x, y) coordinates. "
            "Be brief and factual. The goal is to give a text-only LLM the necessary information to choose the next action."
        )

    def describe_screen(self, screenshot_path: str) -> str:
        """
        Takes a path to a screenshot and returns a text description of its contents.
        """
        self.logger.info(f"Analyzing screenshot with Qwen-VL model: {screenshot_path}")
        try:
            query = self.processor.from_list_format([
                {'image': screenshot_path},
                {'text': self.prompt},
            ])

            response, _ = self.model.chat(self.processor, query=query, history=None)
            self.logger.info(f"Screen description generated: {response}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to describe screen with Qwen-VL model: {e}", exc_info=True)
            return "Error: Could not analyze the screen."
