import datetime, cv2
import numpy as np
from numpy import ndarray
from typing import Optional
import logging

logging.basicConfig(
    filename='typego.log',
    filemode='a',  # 'a' for append, 'w' for overwrite
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(*args):
    logging.info(*args)

def log_warning(*args):
    logging.warning(*args)

def log_error(*args, raise_error=False):
    if raise_error:
        raise ValueError(*args)
    logging.error(*args)

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal: str) -> str:
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

# def quaternion_to_rpy(qx, qy, qz, qw) -> ndarray:
#     """
#     Convert quaternion (qx, qy, qz, qw) to roll, pitch, and yaw (RPY) angles in radians.
#     """
#     # Roll (x-axis rotation)
#     sinr_cosp = 2 * (qw * qx + qy * qz)
#     cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
#     roll = np.arctan2(sinr_cosp, cosr_cosp)

#     # Pitch (y-axis rotation)
#     sinp = 2 * (qw * qy - qz * qx)
#     if abs(sinp) >= 1:
#         pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
#     else:
#         pitch = np.arcsin(sinp)

#     # Yaw (z-axis rotation)
#     siny_cosp = 2 * (qw * qz + qx * qy)
#     cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
#     yaw = np.arctan2(siny_cosp, cosy_cosp)

#     return np.array([roll, pitch, yaw])

class ImageRecover:
    def __init__(self, K: ndarray, D: ndarray):
        """
        Initialize ImageRecover with camera matrix and distortion coefficients.
        
        :param K: Camera matrix (3x3)
        :param D: Distortion coefficients (1x4 or 4x1)
        """
        self.K = K
        self.D = D
        self._new_K: Optional[ndarray] = None
        self._map1: Optional[ndarray] = None
        self._map2: Optional[ndarray] = None
        self._image_shape: Optional[tuple] = None
    
    def _initialize_maps(self, img_shape: tuple) -> None:
        """Initialize undistortion maps for the given image shape."""
        h, w = img_shape[:2]
        
        # Compute new camera matrix
        self._new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), np.eye(3), balance=0.0
        )
        
        # Initialize undistortion maps
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self._new_K, (w, h), cv2.CV_16SC2
        )
        
        self._image_shape = img_shape
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Recover the image using the camera matrix and distortion coefficients.
        
        :param img: Input image (numpy array)
        :return: Recovered image
        """
        # Validate input
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Check if maps need to be (re)initialized
        current_shape = img.shape[:2]
        if (self._map1 is None or self._map2 is None or self._image_shape != current_shape):
            self._initialize_maps(current_shape)
        
        # Remap the image
        recovered_img = cv2.remap(
            img, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return recovered_img