import sys, os
from concurrent import futures
import json, time
import grpc
from vllm import LLM, SamplingParams

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR, "proto"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

def load_model():
    """Load model with vLLM for optimized inference"""
    path = 'meta-llama/Llama-3.2-3B-Instruct'
    
    # vLLM configuration for optimal performance
    # disable_custom_all_reduce helps with multiprocessing compatibility
    llm = LLM(
        model=path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_custom_all_reduce=True,  # Fix multiprocessing issue
        # Use v0 engine if v1 causes issues
        # use_v2_block_manager=False,
    )
    
    return llm

class LLMService(hyrch_serving_pb2_grpc.LLMServiceServicer):
    def __init__(self, port):
        self.llm = load_model()
        self.port = port
        
        # Warm up with a dummy request
        _ = self.llm.generate(["Warm up."], SamplingParams(max_tokens=1))
        print(f"LLM service initialized on port {port}")

    def Detect(self, request, context) -> hyrch_serving_pb2.DetectResponse:
        info = json.loads(request.json_data)

        if 'prompt' not in info:
            info['result'] = 'No prompt provided'
            return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

        prompt = info['prompt']
        max_new_tokens = info.get('max_new_tokens', 1024)
        temperature = info.get('temperature', 0.1)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.95,
        )
        
        # Generate response
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        inference_time = time.time() - start_time
        
        # Extract generated text
        response = outputs[0].outputs[0].text
        info['result'] = response
        info['inference_time_seconds'] = inference_time
        
        print(f"Generated {len(outputs[0].outputs[0].token_ids)} tokens in {inference_time:.3f}s")
        
        return hyrch_serving_pb2.DetectResponse(json_data=json.dumps(info))

def serve(port, stop_event):
    print(f"LLM service at port {port} [STARTING]")
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    hyrch_serving_pb2_grpc.add_LLMServiceServicer_to_server(LLMService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"LLM service at port {port} [RUNNING]")

    if stop_event is None:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"LLM service at port {port} [STOPPED]")
            server.stop(0)
    else:
        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"LLM service at port {port} [STOPPED]")
            server.stop(0)

if __name__ == "__main__":
    # Must use spawn method for vLLM compatibility
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    serve(50055, None)