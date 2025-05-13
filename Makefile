.PHONY: docker_stop, docker_start, docker_remove, docker_open, docker_build, build

build:
	colcon build
	if [ -f ./install/typego/lib/typego/webui ]; then \
		sed -i '1s|^#!.*|#!'"$$(which python)"'|' ./install/typego/lib/typego/webui; \
	else \
		echo "Warning: ./install/typego/lib/typego/webui not found. Skipping shebang update."; \
	fi

docker_stop:
	@echo "=> Stopping go2-sdk..."
	@-docker stop -t 0 go2-sdk > /dev/null 2>&1
	@-docker rm -f go2-sdk > /dev/null 2>&1

docker_start:
	@make docker_stop
	@echo "=> Starting go2-sdk..."
	docker run -td --privileged --net=host \
    	--name="go2-sdk" \
		--env-file ./docker/.env \
		go2-sdk:0.1

docker_remove:
	@echo "=> Removing go2-sdk..."
	@-docker image rm -f go2-sdk:0.1  > /dev/null 2>&1
	@-docker rm -f go2-sdk > /dev/null 2>&1

docker_open:
	@echo "=> Opening bash in go2-sdk..."
	@docker exec -it go2-sdk bash

docker_build:
	@echo "=> Building go2-sdk..."
	@make docker_stop
	@make docker_remove
	@echo -n "=>"
	docker build -t go2-sdk:0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make docker_start

run_slam:
	ros2 launch typego slam_sync.launch.py rviz:=true

save_map:
	ros2 service call /slam_toolbox/serialize_map slam_toolbox/SerializePoseGraph "{filename: '/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/yecl-lab'}"