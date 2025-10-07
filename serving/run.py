import os
import sys
import signal
import uvicorn
import multiprocessing
import time

# === Import AFTER start_method set ===
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

from vision_service import SERVICE_INFO  # safe import after spawn set
EDGE_SERVICE_PORT = int(os.environ.get("EDGE_SERVICE_PORT", "50049"))


def start_worker_service(stop_event):
    """Spawn subprocesses for each gRPC worker."""
    processes = []

    for service in SERVICE_INFO:
        name = service["name"]
        ports = service.get("port", [])
        if not ports:
            continue

        # Lazy import — each process loads CUDA context independently
        if name == "yolo":
            from yolo_service import serve as serve_fn
        elif name == "clip":
            from clip_service import serve as serve_fn
        elif name == "vlm":
            from vlm_service import serve as serve_fn
        elif name == "llm":
            from llm_service import serve as serve_fn
        else:
            continue

        for port in ports:
            p = multiprocessing.Process(target=serve_fn, args=(port, stop_event), daemon=True)
            p.start()
            processes.append(p)

    return processes


def cleanup(_signum, _frame, processes, stop_event):
    print("🧹 Shutting down services...")
    stop_event.set()

    # Give processes some time to stop gracefully
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            print(f"⚠️ Force killing {p.name} (pid={p.pid})")
            p.kill()

    print("✅ All worker services stopped.")
    os._exit(0)


def main():
    stop_event = multiprocessing.Event()
    processes = start_worker_service(stop_event)

    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: cleanup(s, f, processes, stop_event))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(s, f, processes, stop_event))

    # Run Quart/ASGI app in the main process
    from vision_service import app
    try:
        uvicorn.run(app, host="0.0.0.0", port=EDGE_SERVICE_PORT)
    except OSError as e:
        if e.errno == 98:
            print(f"❌ Port {EDGE_SERVICE_PORT} already in use — choose another or kill old process.")
        else:
            raise
    finally:
        cleanup(None, None, processes, stop_event)


if __name__ == "__main__":
    main()
