from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager

import json, os, time
from json import JSONEncoder
import asyncio, aiohttp, threading
import numpy as np
from typing import Any, Optional

from typego.utils import print_t
from typego.robot_info import RobotInfo

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")
# EDGE_SERVICE_IP = "localhost"
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

FONT = ImageFont.truetype(os.path.join(CURRENT_DIR, "resource/Roboto-Medium.ttf"), size=36)

class ObjectInfo:
    def __init__(self, name: str, x, y, w, h, depth: Optional[float] = None):
        self.name: str = name
        self.x: float = float(x)
        self.y: float = float(y)
        self.w: float = float(w)
        self.h: float = float(h)
        self.depth: Optional[float] = float(depth) if depth is not None else None

    # -------- JSON support --------
    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        base_info = {
            "name": self.name,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "w": round(self.w, 2),
            "h": round(self.h, 2),
        }
        if self.depth is not None:
            base_info["dist"] = round(self.depth, 2)
        return base_info

    def to_json(self) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict())

    # -------- Python builtins --------
    def __repr__(self) -> str:
        return f"ObjectInfo({self.to_dict()})"

    def __getitem__(self, key: str | int):
        if isinstance(key, int):  # index-style
            mapping = [self.name, self.x, self.y, self.w, self.h, self.depth]
            return mapping[key]
        elif isinstance(key, str):  # dict-like
            if key == "size":
                return [self.w, self.h]
            if key == "dist":
                return self.depth
            if hasattr(self, key):
                return getattr(self, key)
            raise KeyError(key)
        else:
            raise TypeError("Key must be str or int")

    def __setitem__(self, key: str | int, value):
        if isinstance(key, int):
            attrs = ["name", "x", "y", "w", "h", "depth"]
            setattr(self, attrs[key], value)
        elif isinstance(key, str):
            if key in {"size", "dist"}:
                raise KeyError(f"{key} is derived and cannot be directly set.")
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(key)
        else:
            raise TypeError("Key must be str or int")

    def get(self, key: str, default=None):
        """Dict-like get method."""
        try:
            return self[key]
        except (KeyError, TypeError, IndexError):
            return default
    
class ObservationEncoder(JSONEncoder):
    """Custom JSON encoder for ObjectInfo class"""
    def default(self, obj):
        if isinstance(obj, ObjectInfo):
            return obj.to_dict()
        elif isinstance(obj, (np.float32, np.float64, float)):
            return round(float(obj), 2)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

"""
Access the YOLO service through http.
"""
class YoloClient():
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info
        self.service_url = 'http://{}:{}/process'.format(EDGE_SERVICE_IP, EDGE_SERVICE_PORT)
        self.target_image_size = (640, 360)
        self._latest_result_lock = asyncio.Lock()
        self._latest_result = None
        self.frame_id = 0
        self.frame_id_lock = asyncio.Lock()
        print_t(f"[Y] YoloClient initialized with service url: {self.service_url}")

    @property
    def latest_result(self) -> tuple[Image.Image, list[ObjectInfo]] | None:
        result = self._latest_result
        if result is None:
            return None
        image, objects = result
        # shallow copy of list to decouple from async updates
        return (image, list(objects))

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    @staticmethod
    def plot_results_ps(image: Image.Image, object_list: list[ObjectInfo]) -> Image.Image:
        if len(object_list) == 0:
            return image

        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)

        draw = ImageDraw.Draw(image)
        w, h = image.size

        for obj in object_list:
            x1 = str_float_to_int(obj.x - obj.w / 2, w)
            y1 = str_float_to_int(obj.y - obj.h / 2, h)
            x2 = str_float_to_int(obj.x + obj.w / 2, w)
            y2 = str_float_to_int(obj.y + obj.h / 2, h)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='#00FFFF', width=6)

            # Draw label and depth
            label = f"{obj.name}"
            if obj.depth is not None:
                label += f" ({obj.depth:.2f}m)"

            # label += f" ({obj.x:.2f}, {obj.y:.2f})"

            draw_y = y1 - 40 if y1 - 40 > 0 else y2 + 10
            draw.text((x1, draw_y), label, fill='red', font=FONT)

        return image
    
    @staticmethod
    def cc_to_ps(result: list) -> list[ObjectInfo]:
        return [
            ObjectInfo(
                name=obj['name'],
                x=(obj['box']['x1'] + obj['box']['x2']) / 2,
                y=(obj['box']['y1'] + obj['box']['y2']) / 2,
                w=obj['box']['x2'] - obj['box']['x1'],
                h=obj['box']['y2'] - obj['box']['y1'],
                depth=obj['depth'] if 'depth' in obj else None
            )
            for obj in result
        ]

    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, form_data, timeout_seconds=3):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            # The ClientSession now uses the defined timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(service_url, data=form_data) as response:
                    if response.status != 200:
                        print_t(f"[Y] Invalid response status: {response.status}")
                        response.raise_for_status()  # Optional: raises exception for 4XX/5XX responses
                    yield response
        except aiohttp.ServerTimeoutError:
            print_t(f"[Y] Timeout error when connecting to {service_url}")

    async def detect(self, image: Image.Image, conf=0.3):
        # Prepare image and config while not holding the lock
        config = {
            'robot_info': self.robot_info.robot_id,
            'service_type': 'yolo3d',
            'tracking_mode': False,
            'image_id': 0,
            'conf': conf
        }
        image_bytes = YoloClient.image_to_bytes(image.resize(self.target_image_size))

        async with self.frame_id_lock:
            self.frame_id += 1
            config['image_id'] = self.frame_id
            
            form_data = aiohttp.FormData()
            form_data.add_field('image', image_bytes, filename='frame.webp', content_type='image/webp')
            form_data.add_field('json_data', json.dumps(config), content_type='application/json')

        try:
            async with YoloClient.get_aiohttp_session_response(self.service_url, form_data) as response:
                data = await response.text()
                json_results = json.loads(data)
        except json.JSONDecodeError:
            print_t(f"[Y] Invalid json results: {data}")
            return
        except Exception as e:
            print_t(f"[Y] Request failed: {str(e)}")
            return

        if 'image_id' not in json_results:
            print_t(f"[Y] Missing image_id in results: {json_results}")
            return
        
        async with self._latest_result_lock:
            self._latest_result = (image, self.cc_to_ps(json_results["result"]))