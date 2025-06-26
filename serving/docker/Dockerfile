FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
RUN apt update
RUN pip install grpcio-tools quart uvicorn open_clip_torch

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install ultralytics

RUN apt-get update && apt-get install -y build-essential
RUN pip install transformers einops accelerate bitsandbytes flash-attn

# copy the contents of the current project to /workspace
COPY .. /workspace

# set the working directory to /workspace
WORKDIR /workspace

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./run.py"]