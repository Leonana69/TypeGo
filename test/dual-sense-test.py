import pyrealsense2 as rs
import numpy as np
import cv2, time
from collections import deque


def stack_frames(color1, color2, depth1, depth2):
    K1 = np.array([[606.1963500976562,   0.0, 330.197021484375],
               [  0.0, 606.2576904296875, 248.43824768066406],
               [  0.0,   0.0,   1.0]], dtype=np.float32)

    K2 = K1.copy()
    R = cv2.Rodrigues(np.array([-np.deg2rad(41), 0, 0], dtype=np.float32))[0]
    t = np.array([[0.0, 0.0, 0.06]], dtype=np.float32).T

    t1 = time.time()
    # Assume planar scene at average depth
    avg_depth = np.median(depth2[depth2 > 0]) if np.any(depth2 > 0) else 1000.0

    # Compute homography to warp camera2 to camera1's view
    n = np.array([[0], [0], [1]], dtype=np.float32)
    H = K1 @ (R + (t @ n.T) / avg_depth) @ np.linalg.inv(K2)

    # Warp camera2 to camera1's viewpoint
    h1, w1 = color1.shape[:2]
    color2_warped = cv2.warpPerspective(color2, H, (w1, h1*2 + 600), flags=cv2.INTER_LINEAR)
    depth2_warped = cv2.warpPerspective(depth2, H, (w1, h1*2 + 600), flags=cv2.INTER_NEAREST)

    color2_warped = color2_warped[h1 + 0:, :]
    depth2_warped = depth2_warped[h1 + 0:, :]

    # Stack vertically: camera1 on top, warped camera2 on bottom
    result_color = np.vstack((color1, color2_warped))
    result_depth = np.vstack((depth1, depth2_warped))
    print("Homography computation and warping time:", time.time() - t1)
    return result_color, result_depth

def test_dual_d435i():
    print("Testing dual Intel D435i setup...")

    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Number of devices found: {len(devices)}")

    if len(devices) < 2:
        print("❌ Need at least two D435i cameras connected.")
        return False

    serials = []
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        sn = dev.get_info(rs.camera_info.serial_number)
        print(f"Device {i}: {name}, Serial: {sn}")
        serials.append(sn)

    pipelines = []
    configs = []

    for i, sn in enumerate(serials[:2]):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(sn)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipelines.append(pipeline)
        configs.append(config)

    try:
        print("Starting both pipelines...")
        for p, c in zip(pipelines, configs):
            p.start(c)
        
        # Let cameras warm up
        print("Warming up cameras...")
        for _ in range(30):
            for pipeline in pipelines:
                pipeline.wait_for_frames()
        
        print("✅ Both cameras started successfully!")

        align_to = rs.stream.color
        align = rs.align(align_to)
        hole_filling = rs.hole_filling_filter()

        print("Press 'q' to quit.")
        
        # Frame buffers for each camera
        frame_buffers = [deque(maxlen=10), deque(maxlen=10)]
        max_time_diff_ms = 33  # ~1 frame at 30fps
        
        frame_count = 0

        while True:
            # Collect frames from both cameras into buffers
            for i, pipeline in enumerate(pipelines):
                try:
                    frames = pipeline.poll_for_frames()
                    if frames:
                        frame_buffers[i].append(frames)
                except:
                    pass
            
            # Try to find matching frames
            best_match = None
            best_time_diff = float('inf')
            
            for frames0 in frame_buffers[0]:
                for frames1 in frame_buffers[1]:
                    time_diff = abs(frames0.get_timestamp() - frames1.get_timestamp())
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_match = (frames0, frames1)
            
            # Process if we found a good match
            if best_match and best_time_diff < max_time_diff_ms:
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"✓ Time diff: {best_time_diff:.2f}ms")
                
                colored = []
                depth = []
                raw_depth = []
                
                for frames in best_match:
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    depth_frame = hole_filling.process(depth_frame)

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    raw_depth.append(depth_image)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )

                    colored.append(color_image)
                    depth.append(depth_colormap)

                if len(colored) == 2 and len(depth) == 2:
                    combined_colored, combined_depth = stack_frames(
                        colored[1], colored[0],
                        raw_depth[1], raw_depth[0]
                    )
                    cv2.imshow("Dual D435i Streams (Color | Depth)", combined_colored)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(combined_depth, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow("Dual D435i Streams (Depth Colormap)", depth_colormap)
                
                # Remove used frames from buffers
                if frames0 in frame_buffers[0]:
                    frame_buffers[0].remove(frames0)
                if frames1 in frame_buffers[1]:
                    frame_buffers[1].remove(frames1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"⚠️ Error during streaming: {e}")
    finally:
        print("Stopping pipelines...")
        for p in pipelines:
            p.stop()
        cv2.destroyAllWindows()
        print("✅ Clean exit.")

    return True


if __name__ == "__main__":
    test_dual_d435i()