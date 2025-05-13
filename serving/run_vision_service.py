import multiprocessing
import signal
import uvicorn
import os, sys

from vision_service import SERVICE_INFO
from yolo_service import serve as yolo_serve
from clip_service import serve as clip_serve

EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))

def start_worker_service(stop_event):
    processes = []
    for service in SERVICE_INFO:
        if service["name"] == "yolo":
            for port in service["port"]:
                process = multiprocessing.Process(target=yolo_serve, args=(port, stop_event))
                process.start()
                processes.append(process)
        elif service["name"] == "clip":
            for port in service["port"]:
                process = multiprocessing.Process(target=clip_serve, args=(port, stop_event))
                process.start()
                processes.append(process)
    return processes

def main():
    stop_event = multiprocessing.Event()
    processes = start_worker_service(stop_event)

    def cleanup(_signalnum, _frame):
        print("Shutting down YOLO services...")
        stop_event.set()
        for p in processes:
            p.join()
        print("Shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    from vision_service import app
    uvicorn.run(app, host="0.0.0.0", port=EDGE_SERVICE_PORT)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    main()
