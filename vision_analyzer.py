import logging
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2ForConditionalGeneration # Using the latest Qwen2

class VisionAnalyzer:
    """
    A class to analyze screenshots using a vision-language model (VLM).
    Its primary role is to describe the contents of the screen for another agent.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing VisionAnalyzer on device: {self.device}")

        self.model = Qwen2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt = (
            "You are a helpful UI assistant. Your task is to analyze a screenshot of a phone screen "
            "and provide a detailed, structured description of its contents. "
            "Identify all interactive elements like buttons, text fields, icons, and links. "
            "For each element, describe its purpose and provide its approximate coordinates (e.g., top-left, center, bottom-right). "
            "The goal is to give a text-only LLM enough information to decide on the next action. "
            "Be concise but comprehensive."
        )

    def describe_screen(self, screenshot_path: str) -> str:
        """
        Takes a path to a screenshot and returns a text description of its contents.
        """
        self.logger.info(f"Analyzing screenshot: {screenshot_path}")
        try:
            image = Image.open(screenshot_path)

            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": [{"type": "image", "image": image}]}
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.processor([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.logger.info(f"Screen description generated: {response}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to describe screen: {e}", exc_info=True)
            return "Error: Could not analyze the screen."
