.PHONY: docker_stop, docker_start, docker_remove, docker_open, docker_build

# Set JETSON=1 to build/run the Jetson Dockerfile variant
JETSON ?= 0

GPU_DEVICES=0
GPU_OPTIONS=$(shell if [ -f /proc/driver/nvidia/version ]; then echo "--gpus all -e CUDA_VISIBLE_DEVICES=$(GPU_DEVICES)"; else echo ""; fi)

DOCKERFILE = ./docker/Dockerfile
IMAGE = vision-service:0.1
CONTAINER_NAME = vision-service

ifeq ($(JETSON),1)
	DOCKERFILE = ./docker/Dockerfile-jetson
	IMAGE = vision-service-jetson:0.1
	CONTAINER_NAME = vision-service-jetson
	GPU_OPTIONS = --gpus=all --runtime=nvidia
endif

docker_start:
	@make docker_stop
	@echo "=> Starting $(CONTAINER_NAME)..."
	docker run -td --privileged --net=host $(GPU_OPTIONS) --ipc=host \
		--env-file ./docker/.env \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		--name="$(CONTAINER_NAME)" $(IMAGE)

docker_stop:
	@echo "=> Stopping $(CONTAINER_NAME)..."
	@-docker stop -t 0 $(CONTAINER_NAME) > /dev/null 2>&1
	@-docker rm -f $(CONTAINER_NAME) > /dev/null 2>&1

docker_remove:
	@echo "=> Removing $(CONTAINER_NAME)..."
	@-docker image rm -f $(IMAGE)  > /dev/null 2>&1
	@-docker rm -f $(CONTAINER_NAME) > /dev/null 2>&1

docker_open:
	@echo "=> Opening bash in $(CONTAINER_NAME)..."
	@docker exec -it $(CONTAINER_NAME) bash

docker_build:
	@echo "=> Building $(IMAGE) using $(DOCKERFILE)..."
	@make docker_stop
	@make docker_remove
	@echo -n "=>"
	docker build -t $(IMAGE) -f $(DOCKERFILE) .
	@echo -n "=>"
	@make docker_start