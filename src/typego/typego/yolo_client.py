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
EDGE_SERVICE_PORT = os.environ.get("EDGE_SERVICE_PORT", "50049")

FONT = ImageFont.truetype(os.path.join(CURRENT_DIR, "resource/Roboto-Medium.ttf"), size=36)

class ObjectInfo:
    def __init__(self, name: str, center_x, center_y, width, height, depth: Optional[float] = None):
        self.name: str = name
        self.cx: float = float(center_x)
        self.cy: float = float(center_y)
        self.w: float = float(width)
        self.h: float = float(height)
        self.depth: Optional[float] = float(depth) if depth is not None else None

    # -------- JSON support --------
    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        base_info = {
            "name": self.name,
            "bbox": [round(self.cx, 2), round(self.cy, 2), round(self.w, 2), round(self.h, 2)],
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
            mapping = [self.name, self.cx, self.cy, self.w, self.h, self.depth]
            return mapping[key]
        elif isinstance(key, str):  # dict-like
            if key == "bbox":
                return [self.cx, self.cy, self.w, self.h]
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
            attrs = ["name", "cx", "cy", "w", "h", "depth"]
            setattr(self, attrs[key], value)
        elif isinstance(key, str):
            if key == "bbox":
                if len(value) != 4:
                    raise ValueError("bbox must be a list of 4 values [cx, cy, w, h]")
                self.cx, self.cy, self.w, self.h = value
            elif key in {"size", "dist", "bbox"}:
                raise KeyError(f"{key} is derived and cannot be directly set.")
            elif hasattr(self, key):
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
    def plot_results_ps(image: Image.Image, object_list: list["ObjectInfo"]) -> Image.Image:
        if not object_list:
            return image

        # --- Scale image by 2x ---
        w, h = image.size
        new_size = (w * 2, h * 2)
        image = image.resize(new_size, Image.LANCZOS)
        w, h = new_size

        draw = ImageDraw.Draw(image)

        def scale_coord(cx, cy, bw, bh, width, height):
            """Convert normalized center coords to pixel box corners."""
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            return x1, y1, x2, y2

        def get_text_size(text, font):
            """Return (width, height) for text with the given font (Pillow ≥10 compatible)."""
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height

        for obj in object_list:
            x1, y1, x2, y2 = scale_coord(obj.cx, obj.cy, obj.w, obj.h, w, h)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="#00FFFF", width=6)

            # Prepare label text
            label = obj.name
            if getattr(obj, "depth", None) is not None:
                label += f" ({obj.depth:.2f}m)"

            # Determine label position
            text_w, text_h = get_text_size(label, FONT)
            box_h = y2 - y1
            box_w = x2 - x1

            # If box is tall enough, draw text inside top-left corner
            if box_h > text_h * 1.5 and box_w > text_w + 10:
                text_x, text_y = x1 + 5, y1 + 5
                bg_y1, bg_y2 = text_y - 2, text_y + text_h + 2
            else:
                # Otherwise draw above the box (or below if near top edge)
                text_y = y1 - text_h - 8 if y1 - text_h - 8 > 0 else y2 + 8
                text_x = x1
                bg_y1, bg_y2 = text_y - 2, text_y + text_h + 2

            # Background rectangle for better readability
            bg_x1, bg_x2 = text_x - 2, text_x + text_w + 4
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="black")

            # Draw text label
            draw.text((text_x, text_y), label, fill="red", font=FONT)

        return image

    @staticmethod
    def cc_to_ps(result: list) -> list[ObjectInfo]:
        return [
            ObjectInfo(
                name=obj['name'],
                center_x=(obj['box']['x1'] + obj['box']['x2']) / 2,
                center_y=(obj['box']['y1'] + obj['box']['y2']) / 2,
                width=obj['box']['x2'] - obj['box']['x1'],
                height=obj['box']['y2'] - obj['box']['y1'],
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

    async def detect(self, image: Image.Image, depth: Optional[np.ndarray] = None, conf=0.3):
        # Prepare image and config while not holding the lock
        config = {
            'robot_info': self.robot_info.robot_id,
            'service_type': 'yolo',
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
        
        results = json_results.get('result', [])
        if depth is not None and len(depth.shape) == 2:
            H, W = depth.shape
            for obj in results:
                box = obj['box']
                # Convert normalized coordinates [0,1] → pixel coordinates
                x1, y1 = int(box['x1'] * W), int(box['y1'] * H)
                x2, y2 = int(box['x2'] * W), int(box['y2'] * H)

                # Ensure bounds are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                roi = depth[y1:y2, x1:x2]
                if roi.size > 0:
                    valid_depths = roi[roi > 0]  # filter invalid pixels
                    if valid_depths.size > 0:
                        median_depth_mm = np.median(valid_depths)
                        obj['depth'] = float(median_depth_mm / 1000.0)
                    else:
                        obj['depth'] = None
                else:
                    obj['depth'] = None
        else:
            for obj in results:
                obj['depth'] = None

        # print_t(f"[Y] Detection results for image_id {json_results['image_id']}: {json_results.get('result', [])}")
        async with self._latest_result_lock:
            self._latest_result = (image, self.cc_to_ps(json_results["result"]))