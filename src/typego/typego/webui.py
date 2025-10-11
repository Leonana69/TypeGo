import io, time, json, os
from flask import Flask, Response, render_template, request, jsonify
from threading import Thread

from typego.robot_info import RobotInfo
from typego.utils import print_t
from typego.llm_controller import LLMController
from typego.frontend_message import try_get

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

class TypeGo:
    def __init__(self, robot_info: RobotInfo):
        self.llm_controller = LLMController(robot_info)
        self.running = True
        self.app = Flask(__name__, 
                        template_folder=os.path.join(CURRENT_DIR, 'resource'))
        
        print_t("template_folder:", self.app.template_folder)
        self.setup_routes()

    def setup_routes(self):
        """Sets up the Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Handle chat messages and stream responses using SSE."""
            data = request.get_json()
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({'type': 'text', 'content': '[WARNING] Empty command!'})
            
            if user_message == "exit":
                self.running = False
                return jsonify({'type': 'text', 'content': 'Shutting down...'})
            
            # Send instruction to LLM
            task_id = self.llm_controller.put_instruction(user_message)
            
            def generate():
                # Send initial acknowledgment
                yield f"data: {json.dumps({'type': 'text', 'content': 'Okay! Working on it...'})}\n\n"
                
                # Stream messages from the queue as they arrive
                while True:
                    msg = try_get(timeout=3.0, task_id=task_id)
                    if msg is None:
                        break
                    elif msg == "":
                        continue
                    
                    # print_t(f"[UI] New message: {msg}")
                    msg_str = str(msg)
                    
                    # Check if message contains an image (base64 encoded)
                    if '<img src="data:image/' in msg_str:
                        # For images, we need to ensure JSON encoding doesn't break the HTML
                        response_data = json.dumps({'type': 'image', 'content': msg_str}, ensure_ascii=False)
                        yield f"data: {response_data}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': msg_str})}\n\n"
                
                # Send end signal
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        @self.app.route('/robot-pov/')
        def video_feed_pov():
            """Stream robot POV video feed."""
            return Response(
                self.generate_mjpeg_stream('pov'),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/robot-map/')
        def video_feed_map():
            """Stream robot map video feed."""
            return Response(
                self.generate_mjpeg_stream('map'),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/health')
        def health():
            """Health check endpoint."""
            return jsonify({'status': 'running', 'robot': self.running})

    def generate_mjpeg_stream(self, source: str):
        """Generate MJPEG stream for video feeds."""
        while self.running:
            if source == 'pov':
                frame = self.llm_controller.fetch_robot_pov()
            elif source == 'map':
                frame = self.llm_controller.fetch_robot_map()
            else:
                frame = None
            
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
        """Start the TypeGo system with Flask server."""
        # Start the LLM controller
        self.llm_controller.start_controller()

        # Start the Flask server
        print_t("[TypeGo] Starting Flask server on http://0.0.0.0:50000")
        self.app.run(host='127.0.0.1', port=50000, debug=False, threaded=True)
        
        # When Flask stops, stop the LLM controller
        self.llm_controller.stop_controller()

def main():
    with open(os.path.join(CURRENT_DIR, 'config/robot_info.json'), 'r') as f:
        typego = TypeGo(RobotInfo.from_dict(json.load(f)))
        typego.run()

if __name__ == '__main__':
    main()