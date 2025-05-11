import datetime, cv2
import numpy as np
from numpy import ndarray
from typego.skill_item import SKILL_RET_TYPE
from PIL import Image

def get_current_time() -> str:
    """
    Get the current time in the format HH:MM:SS
    """
    return datetime.datetime.now().strftime('%H:%M:%S')

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def evaluate_value(s: str) -> SKILL_RET_TYPE:
    if s.lstrip('-').isdigit():  # Check for negative integers
        return int(s)
    elif s.lstrip('-').replace('.', '', 1).isdigit():  # Check for negative floats
        return float(s)
    elif s == 'True':
        return True
    elif s == 'False':
        return False
    elif s == 'None' or len(s) == 0:
        return None
    else:
        if not (s.startswith("'") and s.endswith("'")):
            return f"'{s}'"
        return s
    
def quaternion_to_rpy(qx, qy, qz, qw) -> ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to roll, pitch, and yaw (RPY) angles in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

class ImageRecover:
    def __init__(self, K: ndarray, D: ndarray):
        self.K = K
        self.D = D
        self.new_K = None
        self.map1 = None
        self.map2 = None

    def process(self, img: cv2.Mat) -> cv2.Mat:
        """
        Recover the image using the camera matrix and distortion coefficients.
        
        :param img: Input image
        :return: Recovered image
        """
        # Get the dimensions of the image
        dim1 = img.shape[:2][::-1]
        
        # Compute new camera matrix
        if self.new_K is None:
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, dim1, np.eye(3), balance=0.0
            )
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K, dim1, cv2.CV_16SC2
            )
        
        # Remap the image
        recovered_img = cv2.remap(
            img, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return recovered_img
    
def slam_map_overlay(image: Image.Image, slam_map: np.ndarray) -> Image.Image:
    """
    Overlay the SLAM map (grayscale, no transparency) on top of the image at a fixed offset.

    :param image: Input PIL Image (RGB or RGBA)
    :param slam_map: SLAM map as a 2D NumPy array (grayscale)
    :return: PIL Image with SLAM map overlaid
    """
    if image is None or slam_map is None:
        return image

    dx = 10
    dy = 10

    # Convert SLAM map to RGB image (from grayscale)
    slam_map_pil = Image.fromarray(slam_map)
    slam_map_pil = slam_map_pil.resize(
        (slam_map_pil.width * 2, slam_map_pil.height * 2), resample=Image.NEAREST
    )

    # Paste slam_map onto image at (dx, dy), cropping if needed
    img_w, img_h = image.size
    map_w, map_h = slam_map_pil.size

    paste_x = max(0, dx)
    paste_y = max(0, dy)
    crop_x = max(0, -dx)
    crop_y = max(0, -dy)
    crop_w = min(map_w - crop_x, img_w - paste_x)
    crop_h = min(map_h - crop_y, img_h - paste_y)

    # Crop the slam map if it would go out of bounds
    cropped_map = slam_map_pil.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    # Paste directly (no mask, no blending)
    image.paste(cropped_map, (paste_x, paste_y))

    return image