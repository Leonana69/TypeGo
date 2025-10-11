import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import socket, os
import io, time, json
import gradio as gr
from flask import Flask, Response
from threading import Thread

from typego.robot_info import RobotInfo
from typego.utils import print_t
from typego.llm_controller import LLMController
from typego.frontend_message import try_get

CURRENT_DIR = get_package_share_directory('typego')

def generate_povs():
    """Generate HTML string for the robot POV."""
    html_content = "<h2>Robot POV</h2><div style='display: flex; gap: 10px;'>"
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

class TypeGo:
    def __init__(self, robot_info: RobotInfo):
        self.llm_controller = LLMController(robot_info)
        self.running = True

        self.ui = gr.Blocks(title="TypeGo")
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
            gr.HTML("""<h1>🐕 TypeGo: Power the Robot Dog with Large Language Model</h1>""")
            gr.HTML(generate_povs())
            self.chat_interface = gr.ChatInterface(
                fn=self.ui_process_message,
                fill_height=False,
                examples=default_sentences,
                type='messages',
            ).queue()

    def ui_process_message(self, message: str, history: list):
        """Handle new user input and also stream queued messages."""
        self.history = history
        if message == "exit":
            self.running = False
            yield gr.ChatMessage(role="assistant", content="Shutting down...")
            return
        elif len(message) == 0:
            yield gr.ChatMessage(role="assistant", content="[WARNING] Empty command!]")
            return
        else:
            # Send instruction to LLM
            self.llm_controller.put_instruction(message)
            yield gr.ChatMessage(role="assistant", content="Okay! Working on it...")

            # Now stream messages from the queue as they arrive
            while True:
                msg = try_get(timeout=1.0)
                if msg == None:
                    break

                # Yield new assistant messages one by one
                print_t(f"[UI] New message: {msg}")
                yield gr.ChatMessage(role="assistant", content=str(msg))

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
        flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()

        # Start the Gradio UI
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        # Wait for the system to stop
        while self.running:
            time.sleep(1)

        # Stop the LLM controller
        self.llm_controller.stop_controller()

def main():
    with open(os.path.join(CURRENT_DIR, 'config/robot_info.json'), 'r') as f:
        typego = TypeGo(RobotInfo.from_dict(json.load(f)))
        typego.run()
