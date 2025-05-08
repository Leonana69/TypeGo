import datetime, cv2
import numpy as np
from numpy import ndarray
from typego.skill_item import SKILL_RET_TYPE

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