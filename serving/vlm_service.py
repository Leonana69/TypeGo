import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json, time
import grpc
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR, "proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

MODEL_TYPE = "ViT-L-14"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_model():
    path = 'OpenGVLab/InternVL3-8B'
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

"""
    gRPC service class.
"""
class VLMService(hyrch_serving_pb2_grpc.VLMServiceServicer):
    def __init__(self, port):
        self.model, self.tokenizer = load_model()
        self.port = port

    @staticmethod
    def bytes_to_image(image_bytes) -> Image.Image:
        return Image.open(BytesIO(image_bytes))

    def parse_request(self, request) -> tuple[Image.Image, dict]:
        info = json.loads(request.json_data)
        image = VLMService.bytes_to_image(request.image_data)

        return image, info
    
    def Detect(self, request, context) -> hyrch_serving_pb2.DetectResponse:
        # print(f"Received Detect request from {context.peer()} on port {self.port}")
        
        # start_time = time.time()
        image, info = self.parse_request(request)

        if 'prompt' not in info:
            info['result'] = 'No prompt provided'

        print(f"Received Detect request {info['prompt']}")

        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # # single-image single-round conversation (单图单轮对话)
        question = info['prompt']
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        info['result'] = response

        # print(f"Detection took {time.time() - start_time} seconds")
        return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

def serve(port, stop_event):
    print(f"VLM service at port {port} [STARTING]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_VLMServiceServicer_to_server(VLMService(port), server)
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
        print(f"VLM service at port {port} [STOPPED]")
        server.stop(0)

if __name__ == "__main__":
    # test the service
    serve(50054 , None)