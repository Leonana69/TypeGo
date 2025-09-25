import os
from enum import Enum
from openai import OpenAI, Stream, ChatCompletion
import cv2
from numpy import ndarray
from PIL import Image
from io import BytesIO
import base64, json, requests
from groq import Groq

class ModelType(Enum):
    GPT4O = "gpt-4o"
    LOCAL_1B = "local-1b"
    PIE_8B = "pie-8b"
    GROQ = "openai/gpt-oss-120b"

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
CHAT_LOG_FILE = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/chat_log.txt"

class LLMWrapper:
    def __init__(self, temperature: float=0.0):
        self.temperature = temperature
        if os.environ.get("OPENAI_API_KEY") is None and os.environ.get("GROQ_API_KEY") is None:
            raise ValueError("Please set OPENAI_API_KEY or GROQ_API_KEY environment variable.")

        if os.environ.get("OPENAI_API_KEY"):
            self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if os.environ.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def request(self, prompt, model_type: ModelType=ModelType.GPT4O, image: cv2.Mat | Image.Image | None=None) -> str | Stream[ChatCompletion.ChatCompletionChunk]:        
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

        if model_type == ModelType.LOCAL_1B:
            json_data = {
                'robot_info': "robot_1",
                'service_type': 'llm',
                'prompt': prompt,
                'max_new_tokens': 10
            }
            http_load = {
                'json_data': (None, json.dumps(json_data))
            }

            response = requests.post(f"http://{EDGE_SERVICE_IP}:{50049}/process", files=http_load)
            response_text = response.json().get("result", "").split('\n\n\n')[0].strip()
        elif model_type == ModelType.GPT4O:
            response = self.gpt_client.responses.create(
                model=model_type.value,
                input=[{"role": "user", "content": content}],
                temperature=self.temperature,
                stream=False
            )
            response_text = response.output_text
        elif model_type == ModelType.GROQ:
            response = self.groq_client.chat.completions.create(
                model=model_type.value,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False
            )
            response_text = response.choices[0].message.content
        elif model_type == ModelType.PIE_8B:
            json_data = {
                'prompt': prompt
            }
            response = requests.post(f"http://{EDGE_SERVICE_IP}:{9000}/generate", json=json_data)
            return response['result']['output']

        with open(CHAT_LOG_FILE, "a") as f:
            # remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt = prompt
            f.write(remove_leading_prompt + "\n---\n")
            f.write(response_text + "\n---\n")

        return response_text