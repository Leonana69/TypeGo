.PHONY: ros_stop, ros_start, ros_remove, ros_open, ros_build, run

GPU_OPTIONS=--gpus all

ros_stop:
	@echo "=> Stopping go2-livox-receiver..."
	@-docker stop -t 0 go2-livox-receiver > /dev/null 2>&1
	@-docker rm -f go2-livox-receiver > /dev/null 2>&1

ros_start:
	@make ros_stop
	@echo "=> Starting go2-livox-receiver..."
	docker run -td --privileged --net=host \
    	--name="go2-livox-receiver" \
		-e ROS_DOMAIN_ID=0 \
    	-e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
		go2-livox-receiver:0.1

ros_remove:
	@echo "=> Removing go2-livox-receiver..."
	@-docker image rm -f go2-livox-receiver:0.1  > /dev/null 2>&1
	@-docker rm -f go2-livox-receiver > /dev/null 2>&1

ros_open:
	@echo "=> Opening bash in go2-livox-receiver..."
	@docker exec -it go2-livox-receiver bash

ros_build:
	@echo "=> Building go2-livox-receiver..."
	@make ros_stop
	@make ros_remove
	@echo -n "=>"
	docker build -t go2-livox-receiver:0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make ros_start

run:
	ros2 run livox_udp_receiver livox_udp_receiver_node