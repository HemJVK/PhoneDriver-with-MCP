import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class VisionAnalyzer:
    """
    A class to analyze screenshots using the SmolVLM2-2.2B-Instruct vision-language model.
    """
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing VisionAnalyzer with SmolVLM on device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        )
        self.prompt = (
            "You are a concise UI description assistant. Your task is to analyze the provided screenshot of a phone screen. "
            "Identify all interactive elements like buttons, text fields, icons, and links. "
            "For each element, describe its purpose and provide its approximate center (x, y) coordinates. "
            "Be brief and factual. The goal is to give a text-only LLM enough information to decide on the next action."
        )

    def describe_screen(self, screenshot_path: str) -> str:
        """
        Takes a path to a screenshot and returns a text description of its contents.
        """
        self.logger.info(f"Analyzing screenshot with SmolVLM model: {screenshot_path}")
        try:
            raw_image = Image.open(screenshot_path)

            # Format the conversation for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(prompt, images=[raw_image], return_tensors="pt").to(self.device, torch.float16)

            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

            # The full output includes the prompt, so we need to decode and then slice it.
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            # SmolVLM output format may differ slightly, so we look for the response part.
            # A robust way is to find the text after the user's prompt.
            description = generated_text.split(self.prompt)[-1].strip()

            self.logger.info(f"Screen description generated: {description}")
            return description

        except Exception as e:
            self.logger.error(f"Failed to describe screen with SmolVLM model: {e}", exc_info=True)
            return "Error: Could not analyze the screen."
