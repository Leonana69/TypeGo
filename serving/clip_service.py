import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json, time
import grpc
import torch
import open_clip

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR, "proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

MODEL_TYPE = "ViT-L-14"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_TYPE, pretrained='openai')
    tokenizer = open_clip.get_tokenizer(MODEL_TYPE)
    model.to(DEVICE)
    return model, preprocess, tokenizer

"""
    gRPC service class.
"""
class ClipService(hyrch_serving_pb2_grpc.ClipServiceServicer):
    def __init__(self, port):
        self.model, self.preprocess, self.tokenizer = load_model()
        self.port = port

    @staticmethod
    def bytes_to_image(image_bytes) -> Image.Image:
        return Image.open(BytesIO(image_bytes))

    def parse_request(self, request) -> tuple[Image.Image, dict]:
        info = json.loads(request.json_data)
        image = ClipService.bytes_to_image(request.image_data)

        return image, info
    
    def Detect(self, request, context) -> hyrch_serving_pb2.DetectResponse:
        # print(f"Received Detect request from {context.peer()} on port {self.port}")
        
        # start_time = time.time()
        image, info = self.parse_request(request)
        print(f"Received Detect request {info['image_id']}")

        if "queries" not in info:
            info["result"] = []
        else:
            queries = info["queries"]
            text = self.tokenizer(queries).to(DEVICE)
            with torch.no_grad(), torch.autocast(DEVICE):
                text_features = self.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            image = self.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.autocast(DEVICE):
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (50.0 * image_features @ text_features.T).softmax(dim=-1)

                info["result"] = text_probs.cpu().float().numpy()[0].tolist()

        # print(f"Detection took {time.time() - start_time} seconds")
        return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

def serve(port, stop_event):
    print(f"Clip service at port {port} [STARTING]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_ClipServiceServicer_to_server(ClipService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    if stop_event is None:
        while True:
            time.sleep(1)

    try:
        # Wait until the event is set
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"YOLO service at port {port} [STOPPED]")
        server.stop(0)

if __name__ == "__main__":
    # test the service
    serve(50052 , None)