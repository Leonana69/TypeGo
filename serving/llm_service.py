import sys, os, gc
from concurrent import futures
from PIL import Image
from io import BytesIO
import json, time
import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as T

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR, "proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    path = 'meta-llama/Llama-3.2-1B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

"""
    gRPC service class.
"""
class LLMService(hyrch_serving_pb2_grpc.LLMServiceServicer):
    def __init__(self, port):
        self.model, self.tokenizer = load_model()
        self.port = port

        prompt = "Warm up."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=1)
    
    def Detect(self, request, context) -> hyrch_serving_pb2.DetectResponse:
        # print(f"Received Detect request from {context.peer()} on port {self.port}")
        info = json.loads(request.json_data)

        if 'prompt' not in info:
            info['result'] = 'No prompt provided'

        print(f"Received Detect request {info['prompt']}")
        prompt = info['prompt']
        max_new_tokens = info.get('max_new_tokens', 1024)
        temperature = info.get('temperature', 0.1)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True, temperature=temperature)
            input_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            info['result'] = response

        # print(f"Detection took {time.time() - start_time} seconds")
        return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

def serve(port, stop_event):
    print(f"LLM service at port {port} [STARTING]")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    hyrch_serving_pb2_grpc.add_LLMServiceServicer_to_server(LLMService(port), server)
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
    serve(50055 , None)