import multiprocessing
import signal
import uvicorn
import os, sys

from yolo_service import serve
YOLO_SERVICE_INFO = { "host": "localhost", "port" : [50050, 50051] }
EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))

def start_yolo_service(stop_event):
    processes = []
    for port in YOLO_SERVICE_INFO["port"]:
        process = multiprocessing.Process(target=serve, args=(port, stop_event))
        process.start()
        processes.append(process)
    return processes

def main():
    stop_event = multiprocessing.Event()
    processes = start_yolo_service(stop_event)

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
    uvicorn.run(app, host="127.0.0.1", port=EDGE_SERVICE_PORT)

if __name__ == "__main__":
    main()
