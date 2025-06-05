import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import sys
print(sys.executable)

import queue
import os, sys
import io, time, json
import gradio as gr
from flask import Flask, Response
from threading import Thread

from typego.robot_info import RobotInfo
from typego.utils import print_t
from typego.llm_controller import LLMController

CURRENT_DIR = get_package_share_directory('typego')

def generate_povs():
    """Generate HTML string for multiple drone POVs."""
    
    html_content = "<h2>Robot POVs</h2><div style='display: flex; gap: 10px;'>"
    html_content += f"""
    <div style="flex: 0 0 70%;">
        <img src="http://localhost:50000/robot-pov/" alt="robot-pov" 
             style="border-radius: 10px; object-fit: contain; width: 100%;">
    </div>
    <div style="flex: 0 0 30%;">
        <img src="http://localhost:50000/robot-map/" alt="robot-map" 
             style="border-radius: 10px; object-fit: contain; width: 100%;">
    </div>
    """
    html_content += "</div>"
    return html_content

class TypeFly:
    def __init__(self, robot_info: RobotInfo):
        self.message_queue = queue.Queue()
        self.llm_controller = LLMController(robot_info, self.message_queue)
        self.running = True

        self.ui = gr.Blocks(title="TypeFly")
        self.setup_ui()

    def setup_ui(self):
        """Sets up the Gradio UI components."""
        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book.",
        ]
        with self.ui:
            gr.HTML("""<h1>🪽 TypeFly: Power the Drone with Large Language Model</h1>""")
            gr.HTML(generate_povs())
            self.chat_interface = gr.ChatInterface(
                fn=self.ui_process_message,
                fill_height=False,
                examples=default_sentences,
                type='messages',
            ).queue()

        # Start a separate thread for streaming assistant messages
        Thread(target=self.stream_from_queue, daemon=True).start()

    def ui_process_message(self, message: str, history: list):
        print_t(f"[S] Receiving task description: {message}, history: {history}")
        self.history = history
        if message == "exit":
            self.running = False
            return gr.ChatMessage(role="assistant", content="Shutting down...")
        elif len(message) == 0:
            return gr.ChatMessage(role="assistant", content="[WARNING] Empty command!]")
        else:
            self.llm_controller.user_instruction(message)
            return gr.ChatMessage(role="assistant", content="Okay! Working on it...")

    def stream_from_queue(self):
        while self.running:
            try:
                msg = self.message_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if isinstance(msg, tuple):  # (image,) or similar
                history = gr.ChatMessage(role="assistant", content=msg)
            elif isinstance(msg, str):
                history = gr.ChatMessage(role="assistant", content=msg)

            print_t(f"[S] Received message: {msg}")


    def generate_mjpeg_stream(self, source: str):
        while self.running:
            if source == 'pov':
                frame = self.llm_controller.fetch_robot_pov()
            elif source == 'map':
                frame = self.llm_controller.fetch_robot_map()
            
            if frame is None:
                time.sleep(1.0 / 30.0)
                continue
                
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        # Start the LLM controller
        self.llm_controller.start_controller()

        # Start the Flask server for video feed
        app = Flask(__name__)
        @app.route('/robot-pov/')
        def video_feed_pov():
            return Response(
                self.generate_mjpeg_stream('pov'), 
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @app.route('/robot-map/')
        def video_feed_map():
            return Response(
                self.generate_mjpeg_stream('map'), 
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()

        # Start the Gradio UI
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        # Wait for the system to stop
        while self.running:
            time.sleep(1)

        # Stop the LLM controller
        self.llm_controller.stop_controller()

"""
Available robot types: ['tello', 'virtual', 'go2']
"""
def main():
    with open(os.path.join(CURRENT_DIR, 'config/robot_info.json'), 'r') as f:
        typefly = TypeFly(RobotInfo.from_dict(json.load(f)))
        typefly.run()
