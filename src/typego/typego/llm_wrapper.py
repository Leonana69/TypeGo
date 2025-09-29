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
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not (openai_key or groq_key):
            raise ValueError("Please set OPENAI_API_KEY or GROQ_API_KEY environment variable.")

        if openai_key:
            self.gpt_client = OpenAI(api_key=openai_key)

        if groq_key:
            self.groq_client = Groq(api_key=groq_key)

    def request(
        self,
        prompt: str,
        model_type: ModelType = ModelType.GPT4O,
        image: cv2.Mat | Image.Image | None = None
    ) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        def encode_image(img) -> str:
            """Convert cv2 or PIL image to base64 JPEG string."""
            if isinstance(img, ndarray):  # cv2
                _, buffer = cv2.imencode(".jpeg", img)
                return base64.b64encode(buffer).decode("utf-8")
            if isinstance(img, Image.Image):  # PIL
                buf = BytesIO()
                img.save(buf, format="JPEG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            raise TypeError("Unsupported image type")

        # Construct content
        content = [{"type": "input_text", "text": prompt}]
        if image is not None:
            image_data = encode_image(image)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            })

        # Route by model type
        if model_type == ModelType.LOCAL_1B:
            payload = {
                "robot_info": "robot_1",
                "service_type": "llm",
                "prompt": prompt,
                "max_new_tokens": 10,
            }
            response = requests.post(
                f"http://{EDGE_SERVICE_IP}:50049/process",
                files={"json_data": (None, json.dumps(payload))}
            )
            response_text = response.json().get("result", "").split("\n\n\n")[0].strip()

        elif model_type == ModelType.GPT4O:
            response = self.gpt_client.responses.create(
                model=model_type.value,
                input=[{"role": "user", "content": content}],
                temperature=self.temperature,
                stream=False,
            )
            response_text = response.output_text

        elif model_type == ModelType.GROQ:
            response = self.groq_client.chat.completions.create(
                model=model_type.value,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
            )
            response_text = response.choices[0].message.content

        elif model_type == ModelType.PIE_8B:
            response = requests.post(
                f"http://{EDGE_SERVICE_IP}:9000/generate",
                json={"prompt": prompt}
            )
            return response.json()["result"]["output"]

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Log request/response
        with open(CHAT_LOG_FILE, "a") as f:
            f.write(prompt + "\n---\n")
            f.write(response_text + "\n---\n")

        return response_text