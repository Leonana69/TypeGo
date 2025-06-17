.PHONY: docker_stop, docker_start, docker_remove, docker_open, docker_build

GPU_OPTIONS=$(shell if [ -f /proc/driver/nvidia/version ]; then echo "--gpus all"; else echo ""; fi)

docker_start:
	@make docker_stop
	@echo "=> Starting vision-service..."
	docker run -td --privileged --net=host $(GPU_OPTIONS) --ipc=host \
		--env-file ./docker/.env \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
    	--name="vision-service" vision-service:0.1

docker_stop:
	@echo "=> Stopping vision-service..."
	@-docker stop -t 0 vision-service > /dev/null 2>&1
	@-docker rm -f vision-service > /dev/null 2>&1

docker_remove:
	@echo "=> Removing vision-service..."
	@-docker image rm -f vision-service:0.1  > /dev/null 2>&1
	@-docker rm -f vision-service > /dev/null 2>&1

docker_open:
	@echo "=> Opening bash in vision-service..."
	@docker exec -it vision-service bash

docker_build:
	@echo "=> Building vision-service..."
	@make docker_stop
	@make docker_remove
	@echo -n "=>"
	docker build -t vision-service:0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make docker_start