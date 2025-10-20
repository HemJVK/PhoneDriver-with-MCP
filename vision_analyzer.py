import logging
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class VisionAnalyzer:
    """
    A class to analyze screenshots using the LLaVA-Llama-3 vision-language model.
    This model is compatible with modern transformers versions and is excellent for UI description.
    """
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-llama3-8b-hf"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing VisionAnalyzer with LLaVA on device: {self.device}")

        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.prompt = (
            "USER: <image>\n"
            "You are a concise UI description assistant. Your task is to analyze the provided screenshot of a phone screen. "
            "Identify all interactive elements like buttons, text fields, icons, and links. "
            "For each element, describe its purpose and provide its approximate center (x, y) coordinates. "
            "Be brief and factual. The goal is to give a text-only LLM enough information to decide on the next action. "
            "ASSISTANT:"
        )

    def describe_screen(self, screenshot_path: str) -> str:
        """
        Takes a path to a screenshot and returns a text description of its contents.
        """
        self.logger.info(f"Analyzing screenshot with LLaVA model: {screenshot_path}")
        try:
            raw_image = Image.open(screenshot_path)

            inputs = self.processor(self.prompt, raw_image, return_tensors="pt").to(self.device, torch.float16)

            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

            # The full output includes the prompt, so we need to decode and then slice it.
            full_response = self.processor.decode(output[0], skip_special_tokens=True)
            description = full_response.split("ASSISTANT:")[1].strip()

            self.logger.info(f"Screen description generated: {description}")
            return description

        except Exception as e:
            self.logger.error(f"Failed to describe screen with LLaVA model: {e}", exc_info=True)
            return "Error: Could not analyze the screen."
