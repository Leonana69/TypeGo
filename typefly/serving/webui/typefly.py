import queue
import os, sys
import io, time, json
import gradio as gr
from flask import Flask, Response
from threading import Thread

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, PROJ_DIR)
from typefly.llm_controller import LLMController
from typefly.utils import print_t
from typefly.robot_info import RobotInfo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_povs():
    """Generate HTML string for multiple drone POVs."""
    
    html_content = "<h2>Robot POVs</h2><div style='display: flex; gap: 10px;'>"
    html_content += f"""
    <div>
        <img src="http://localhost:50000/robot-pov/" alt=robot-pov" 
                style="border-radius: 10px; object-fit: contain; width: 1;">
    </div>
    """
    html_content += "</div>"
    return html_content

class TypeFly:
    def __init__(self, robot_info: RobotInfo):
        self.message_queue = queue.Queue()
        self.llm_controller = LLMController(robot_info, self.message_queue)
        self.system_stop = False

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
            gr.ChatInterface(self.ui_process_message, fill_height=False, examples=default_sentences, type='messages').queue()

    def ui_process_message(self, message: str, history: list):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.system_stop = True
            yield gr.ChatMessage(role="assistant", content="Shutting down...")
        elif len(message) == 0:
            yield gr.ChatMessage(role="assistant", content="[WARNING] Empty command!]")
        else:
            task_thread = Thread(target=self.llm_controller.handle_task, args=(message,))
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple): # (image,)
                    yield gr.ChatMessage(role="assistant", content=msg)
                elif isinstance(msg, str): # "text"
                    if msg == 'end':
                        # Indicate end of the task to Gradio chat
                        print_t(f"[C] Task completed!")
                        return "Command Complete!"
                    
                    if msg.startswith('[LOG]'):
                        complete_response += '\n'
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield gr.ChatMessage(role="assistant", content=complete_response)

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
                
            frame = self.llm_controller.fetch_robot_observation(True)
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
        def video_feed():
            """Route to get video feed for a specific robot."""
            return Response(
                self.generate_mjpeg_stream(), 
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': True, 'use_reloader': False})
        flask_thread.start()

        # Start the Gradio UI
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)

        # Wait for the system to stop
        while not self.system_stop:
            time.sleep(1)

        # Stop the LLM controller
        self.llm_controller.stop_controller()


"""
Available robot types: ['tello', 'virtual', 'go2']
"""
def main():
    with open(os.path.join(CURRENT_DIR, 'robot_info.json'), 'r') as f:
        typefly = TypeFly(RobotInfo.from_dict(json.load(f)))
        typefly.run()