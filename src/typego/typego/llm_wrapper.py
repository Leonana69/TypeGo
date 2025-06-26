import os
from enum import Enum
from openai import OpenAI, Stream, ChatCompletion
import cv2
from numpy import ndarray
from PIL import Image
from io import BytesIO
import base64

class ModelType(Enum):
    GPT4O = "gpt-4o"

CHAT_LOG_FILE = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/chat_log.txt"

class LLMWrapper:
    def __init__(self, temperature: float=0.0):
        self.temperature = temperature
        self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def request(self, prompt, model_type: ModelType=ModelType.GPT4O, stream: bool=False, image: cv2.Mat | Image.Image | None=None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:        
        content = [{
            "type": "input_text",
            "text": prompt
        }]

        if image is not None:
            if isinstance(image, ndarray):
                _, buffer = cv2.imencode(".jpeg", image)
                image_data = base64.b64encode(buffer).decode("utf-8")
            elif isinstance(image, Image.Image):
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            })

        response = self.gpt_client.responses.create(
            model=model_type.value,
            input=[{"role": "user", "content": content}],
            temperature=self.temperature,
            stream=stream,
        )

        with open(CHAT_LOG_FILE, "a") as f:
            # remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt = prompt
            f.write(remove_leading_prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        return response.output_text