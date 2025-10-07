import os
import sys
import json
import logging
from quart import Quart, request, jsonify

# === Path setup ===
PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

from service_manager import ServiceManager

sys.path.append(os.path.join(PROJ_DIR, "proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc


# === App setup ===
app = Quart(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

grpcServiceManager = ServiceManager()

SERVICE_INFO = [
    {"name": "yolo", "host": "localhost", "port": [50050, 50051], "require_image": True},
    {"name": "clip", "host": "localhost", "port": [50052], "require_image": True},
    # {"name": "vlm", "host": "localhost", "port": [50054], "require_image": True},
    {"name": "llm", "host": "localhost", "port": [50055], "require_image": False},
]

# === Stub mapping ===
STUB_MAP = {
    "yolo": hyrch_serving_pb2_grpc.YoloServiceStub,
    "clip": hyrch_serving_pb2_grpc.ClipServiceStub,
    "vlm": hyrch_serving_pb2_grpc.VLMServiceStub,
    "llm": hyrch_serving_pb2_grpc.LLMServiceStub,
}


# === Initialize gRPC Channels once ===
@app.before_serving
async def before_serving():
    for s in SERVICE_INFO:
        grpcServiceManager.add_service(s["name"], s["host"], s["port"])
    await grpcServiceManager._initialize_channels()
    logger.info("✅ All gRPC channels initialized.")


# === Request handler ===
@app.route("/process", methods=["POST"])
async def process():
    try:
        form = await request.form
        json_str = form.get("json_data")
        if not json_str:
            return jsonify(error="Missing json data"), 400

        json_data = json.loads(json_str)
        service_type = json_data.get("service_type")
        robot_info = json_data.get("robot_info")

        if service_type not in STUB_MAP:
            return jsonify(error=f"Unsupported service_type: {service_type}"), 400

        # Determine if service requires image
        service_cfg = next((s for s in SERVICE_INFO if s["name"] == service_type), None)
        if service_cfg and service_cfg.get("require_image"):
            files = await request.files
            if "image" not in files:
                return jsonify(error="Missing image file"), 400
            image_data = files["image"]
            image_bytes = image_data.read()
        else:
            image_bytes = None

        # Get channel
        channel = await grpcServiceManager.get_service_channel(service_type, robot_info)
        if isinstance(channel, str):  # Error string from ServiceManager
            return jsonify(error=f"Channel error: {channel}"), 400

        # Create stub & request
        stub_cls = STUB_MAP[service_type]
        stub = stub_cls(channel)

        req_kwargs = {"json_data": json_str}
        if image_bytes is not None:
            req_kwargs["image_data"] = image_bytes

        response = await stub.Detect(hyrch_serving_pb2.DetectRequest(**req_kwargs))
        return response.json_data

    except json.JSONDecodeError:
        return jsonify(error="Invalid JSON format"), 400
    except KeyError as e:
        return jsonify(error=f"Missing key: {e}"), 400
    except Exception as e:
        logger.exception("Processing error")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
