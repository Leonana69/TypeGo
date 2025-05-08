.PHONY: docker_stop, docker_start, docker_remove, docker_open, docker_build, run, build

GPU_OPTIONS=--gpus all

docker_stop:
	@echo "=> Stopping go2-livox-receiver..."
	@-docker stop -t 0 go2-livox-receiver > /dev/null 2>&1
	@-docker rm -f go2-livox-receiver > /dev/null 2>&1

docker_start:
	@make docker_stop
	@echo "=> Starting go2-livox-receiver..."
	docker run -td --privileged --net=host \
    	--name="go2-livox-receiver" \
		-e docker_DOMAIN_ID=0 \
    	-e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
		go2-livox-receiver:0.1

docker_remove:
	@echo "=> Removing go2-livox-receiver..."
	@-docker image rm -f go2-livox-receiver:0.1  > /dev/null 2>&1
	@-docker rm -f go2-livox-receiver > /dev/null 2>&1

docker_open:
	@echo "=> Opening bash in go2-livox-receiver..."
	@docker exec -it go2-livox-receiver bash

docker_build:
	@echo "=> Building go2-livox-receiver..."
	@make docker_stop
	@make docker_remove
	@echo -n "=>"
	docker build -t go2-livox-receiver:0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make docker_start

build:
	colcon build
	if [ -f ./install/typego/lib/typego/webui ]; then \
		sed -i '1s|^#!.*|#!'"$$(which python)"'|' ./install/typego/lib/typego/webui; \
	else \
		echo "Warning: ./install/typego/lib/typego/webui not found. Skipping shebang update."; \
	fi

run:
	ros2 run livox_udp_receiver livox_udp_receiver_node