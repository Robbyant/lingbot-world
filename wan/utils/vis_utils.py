import numpy as np
import cv2


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, r, alpha=0.6):
    """
    Draw a rounded rectangle with transparency.
    """
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw filled rounded rectangle on overlay
    # Top-left
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
    # Top-right
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
    # Bottom-right
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
    # Bottom-left
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
    
    # Fill centers
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw border (optional, if thickness > 0)
    if thickness > 0:
        pass
    return img

def draw_chevron(img, center, size, direction, color, thickness=3):
    """
    Draw a chevron (V-shape) arrow.
    """
    x, y = center
    offset = size // 2
    
    # Define points based on direction
    if direction == "up":
        # Pointing up: ^
        pts = np.array([
            [x - offset, y + offset // 2],
            [x, y - offset // 2],
            [x + offset, y + offset // 2]
        ], np.int32)
    elif direction == "down":
        # Pointing down: v
        pts = np.array([
            [x - offset, y - offset // 2],
            [x, y + offset // 2],
            [x + offset, y - offset // 2]
        ], np.int32)
    elif direction == "left":
        # Pointing left: <
        pts = np.array([
            [x + offset // 2, y - offset],
            [x - offset // 2, y],
            [x + offset // 2, y + offset]
        ], np.int32)
    elif direction == "right":
        # Pointing right: >
        pts = np.array([
            [x - offset // 2, y - offset],
            [x + offset // 2, y],
            [x - offset // 2, y + offset]
        ], np.int32)
    else:
        return img

    # Draw the polyline
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def draw_wasd_ui(frame, wasd_onehot, position):
    """
    Draw the WASD UI on the frame with active keys highlighted.
    """
    key_size = 40
    spacing = 5
    x, y = position

    keys = {
        "W": (x + key_size + spacing, y, 0),
        "A": (x, y + key_size + spacing, 1),
        "S": (x + key_size + spacing, y + key_size + spacing, 2),
        "D": (x + 2 * (key_size + spacing), y + key_size + spacing, 3),
    }

    for key, (kx, ky, idx) in keys.items():
        # Colors: Blue for active, Dark Grey for inactive
        # Using BGR format for OpenCV
        bg_color = (0, 100, 200) if wasd_onehot[idx] == 1 else (50, 50, 50) # Blue if active
        
        # Draw rounded rectangle background
        frame = draw_rounded_rectangle(frame, (kx, ky), (kx + key_size, ky + key_size), bg_color, -1, r=5, alpha=0.7)

        # Draw key text
        text_color = (255, 255, 255)
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = kx + (key_size - text_size[0]) // 2
        text_y = ky + (key_size + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    return frame

def draw_ijkl_ui(frame, rotation_direction, width, height):
    """
    Draw IJKL rotation control UI (similar to WASD layout).
    Layout:     I
              J K L
    
    I - Up, J - Left, K - Down, L - Right
    Position: Bottom-right corner of the screen
    Keys are always visible, highlighted in orange when active.
    """
    key_size = 40
    spacing = 5
    margin_right = 50
    margin_bottom = 50
    
    x = width - margin_right - 3 * (key_size + spacing) + spacing
    y = height - margin_bottom - 2 * (key_size + spacing) + spacing

    keys = {
        "I": (x + key_size + spacing, y, "up"),
        "J": (x, y + key_size + spacing, "left"),
        "K": (x + key_size + spacing, y + key_size + spacing, "down"),
        "L": (x + 2 * (key_size + spacing), y + key_size + spacing, "right"),
    }

    for key, (kx, ky, direction) in keys.items():
        # Orange for active, Dark Grey for inactive
        bg_color = (200, 100, 0) if rotation_direction == direction else (50, 50, 50)
        
        frame = draw_rounded_rectangle(
            frame, 
            (kx, ky), 
            (kx + key_size, ky + key_size), 
            bg_color, 
            -1, 
            r=5, 
            alpha=0.7
        )

        text_color = (255, 255, 255)
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = kx + (key_size - text_size[0]) // 2
        text_y = ky + (key_size + text_size[1]) // 2
        cv2.putText(
            frame, 
            key, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            thickness, 
            cv2.LINE_AA
        )

    return frame

def draw_rotation_ui(frame, rotation_direction, width, height, mode='arrow'):
    """
    Draw the rotation UI with two modes:
    - 'arrow': Chevron arrows on screen edges (only when rotating)
    - 'keys': IJKL keys layout in bottom-right corner (always visible)
    
    """
    if mode == 'arrow':
    # Arrow mode: only show when there's rotation
        if rotation_direction == 'no':
            return frame

        # Settings
        box_size = 60
        margin = 30  # Distance from screen edge
        bg_color = (0, 0, 0) # Black background
        arrow_color = (255, 255, 255) # White arrow
        alpha = 0.5 # Transparency
        
        # Determine position based on direction
        if rotation_direction == "left":
            center_x = margin + box_size // 2
            center_y = height // 2
        elif rotation_direction == "right":
            center_x = width - margin - box_size // 2
            center_y = height // 2
        elif rotation_direction == "up":
            center_x = width // 2
            center_y = margin + box_size // 2
        elif rotation_direction == "down":
            center_x = width // 2
            center_y = height - margin - box_size // 2
        else:
            return frame

        # Calculate top-left and bottom-right for the box
        pt1 = (center_x - box_size // 2, center_y - box_size // 2)
        pt2 = (center_x + box_size // 2, center_y + box_size // 2)

        # 1. Draw the semi-transparent rounded background
        frame = draw_rounded_rectangle(frame, pt1, pt2, bg_color, -1, r=10, alpha=alpha)

        # 2. Draw the Chevron arrow inside
        frame = draw_chevron(frame, (center_x, center_y), size=int(box_size * 0.5), 
                            direction=rotation_direction, color=arrow_color, thickness=4)
    
    elif mode == 'keys':
        # Keys mode: always show IJKL, highlight when active
        frame = draw_ijkl_ui(frame, rotation_direction, width, height)
    
    return frame

def rotation_matrix_to_euler_angles_opencv(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy < 1e-6:
        z_angle = 0
        y_angle = np.arctan2(-R[2, 0], R[0, 0])
        x_angle = np.arctan2(-R[1, 2], R[1, 1])
    else:
        z_angle = np.arctan2(R[1, 0], R[0, 0])
        y_angle = np.arctan2(-R[2, 0], sy)
        x_angle = np.arctan2(R[2, 1], R[2, 2])
    return z_angle, y_angle, x_angle

def compute_rotation_angles_batch_opencv(c2w_a_batch, c2w_b_batch):
    c2w_a_inv_batch = np.linalg.inv(c2w_a_batch)
    T_rel_batch = np.matmul(c2w_a_inv_batch, c2w_b_batch)
    R_rel_batch = T_rel_batch[:, :3, :3]
    rotation_angles = []
    for R_rel in R_rel_batch:
        z_angle, y_angle, x_angle = rotation_matrix_to_euler_angles_opencv(R_rel)
        rotation_angles.append([z_angle, y_angle, x_angle])
    return np.array(rotation_angles)

def extract_rotation_directions(c2w_poses, threshold=0.005):
    rotation_actions = []
    rotates = compute_rotation_angles_batch_opencv(c2w_poses[:-1], c2w_poses[1:])
    for rotate in rotates:
        z, right, up = rotate
        
        if max(abs(z), abs(right), abs(up)) < threshold:
            rotation_actions.append('no')
        elif max(abs(z), abs(right), abs(up)) == abs(right):
            # Yaw (Left/Right)
            rotation_actions.append('right' if right > 0 else 'left')
        elif max(abs(z), abs(right), abs(up)) == abs(up):
            # Pitch (Up/Down)
            rotation_actions.append('up' if up > 0 else 'down') 
        else:
            rotation_actions.append('no')
    rotation_actions.append('no')
    return rotation_actions

def extract_translation_wasd(c2ws, threshold=0.01):
    """
    Extract WASD actions from translation in c2w poses.
    
    Args:
        c2ws: Array of shape (N, 4, 4)
        threshold: Minimum translation magnitude to register as movement
    
    Returns:
        wasd_actions: Array of shape (N, 4) with one-hot encoded WASD
                     [W, A, S, D] = [forward, left, backward, right]
    """
    # Compute relative transformations
    c2w_a_inv = np.linalg.inv(c2ws[:-1])
    T_rels = np.matmul(c2w_a_inv, c2ws[1:])
    
    wasd_actions = []
    for T_rel in T_rels:
        # Extract translation in camera frame
        right, down, forward = T_rel[:3, -1]
        
        # Check if movement is significant
        if max(abs(right), abs(forward)) < threshold:
            # No movement
            wasd_actions.append([0, 0, 0, 0])
        else:
            # Determine primary direction
            if abs(forward) > abs(right):
                # Forward/Backward dominates
                if forward > 0:
                    wasd_actions.append([1, 0, 0, 0])  # W - forward
                else:
                    wasd_actions.append([0, 0, 1, 0])  # S - backward
            else:
                # Left/Right dominates
                if right > 0:
                    wasd_actions.append([0, 0, 0, 1])  # D - right
                else:
                    wasd_actions.append([0, 1, 0, 0])  # A - left
    
    # Add last frame with no action
    wasd_actions.append([0, 0, 0, 0])
    return np.array(wasd_actions)

def ijkl_onehot_to_direction(ijkl_onehot):
    """
    Convert a single ijkl one-hot array to rotation direction string.
    
    Mapping:
        [1, 0, 0, 0] (i) -> 'up'
        [0, 1, 0, 0] (j) -> 'left'
        [0, 0, 1, 0] (k) -> 'down'
        [0, 0, 0, 1] (l) -> 'right'
        [0, 0, 0, 0] -> 'no'
    """
    idx_to_direction = ['up', 'left', 'down', 'right']
    for i in range(4):
        if ijkl_onehot[i] > 0.5:
            return idx_to_direction[i]
    return 'no'


def visualize_wasd_and_rotation_ui(
    frames: np.ndarray,              # [f, h, w, 3], range [0, 1], RGB format
    c2ws: np.ndarray=None,           # [f, 4, 4]
    wasd_actions: np.ndarray=None,   # [f, 4], one-hot
    ijkl_actions: np.ndarray=None,   # [f, 4], one-hot
    translation_threshold=0.01,
    rotation_threshold=0.005,
    rotation_ui_mode='keys',         # 'arrow' / 'keys'
) -> np.ndarray:                     # [f, h, w, 3], range [0, 1], RGB format
    frames = (frames * 255).astype(np.uint8)
    frames = frames[..., ::-1]       # Convert RGB to BGR for OpenCV

    # Determine wasd_actions
    if wasd_actions is None and c2ws is None:
        raise ValueError("Either wasd_actions or c2ws must be provided.")
    elif wasd_actions is None:
        wasd_actions = extract_translation_wasd(c2ws, translation_threshold)
    
    # Determine rotation_directions: prioritize ijkl_actions over c2ws
    if ijkl_actions is not None:
        rotation_directions = [ijkl_onehot_to_direction(ijkl) for ijkl in ijkl_actions]
    elif c2ws is not None:
        rotation_directions = extract_rotation_directions(c2ws, rotation_threshold)
    else:
        rotation_directions = None
    
    num_frames, height, width, _ = frames.shape
    output_frames = []

    for frame_idx in range(min(num_frames, len(wasd_actions))):
        frame = frames[frame_idx].copy()

        wasd_onehot = wasd_actions[frame_idx]

        # Draw WASD UI
        wasd_position = (50, height - 150)
        frame = draw_wasd_ui(frame, wasd_onehot, wasd_position)

        # Draw Rotation UI
        if rotation_directions is not None:
            rotation_direction = rotation_directions[frame_idx]
            frame = draw_rotation_ui(frame, rotation_direction, width, height, mode=rotation_ui_mode)

        output_frames.append(frame)

    output_frames = np.array(output_frames)[..., ::-1]  # Convert BGR back to RGB
    output_frames = output_frames.astype(np.float32) / 255.0
    
    return output_frames


def rotation_directions_to_onehot(rotation_directions):
    """
    Convert rotation direction strings to one-hot encoded array.
    
    Mapping:
        'up' (i) -> [1, 0, 0, 0]
        'left' (j) -> [0, 1, 0, 0]
        'down' (k) -> [0, 0, 1, 0]
        'right' (l) -> [0, 0, 0, 1]
        'no' -> [0, 0, 0, 0]
    
    Args:
        rotation_directions: List of strings ['up', 'left', 'down', 'right', 'no']
    
    Returns:
        rotation_onehot: Array of shape (N, 4)
    """
    direction_to_onehot = {
        'up': [1, 0, 0, 0],      # i
        'left': [0, 1, 0, 0],    # j
        'down': [0, 0, 1, 0],    # k
        'right': [0, 0, 0, 1],   # l
        'no': [0, 0, 0, 0],
    }
    
    rotation_onehot = []
    for direction in rotation_directions:
        rotation_onehot.append(direction_to_onehot.get(direction, [0, 0, 0, 0]))
    
    return np.array(rotation_onehot)


if __name__ == "__main__":
    import os
    
    translation_threshold = 0.01
    rotation_threshold = 0.005
    
    input_path = '/path/to/examples/00/poses.npy'
    output_dir = os.path.dirname(input_path)
    
    # Load c2w poses
    c2ws = np.load(input_path)
    
    # Extract WASD actions as [n, 4] array
    wasd_actions = extract_translation_wasd(c2ws, translation_threshold).astype(np.float32)
    
    # Extract rotation directions and convert to [n, 4] array
    rotation_directions = extract_rotation_directions(c2ws, rotation_threshold)
    rotation_actions = rotation_directions_to_onehot(rotation_directions).astype(np.float32)
    
    # Save as numpy arrays
    wasd_output_path = os.path.join(output_dir, 'wasd_action.npy')
    rotation_output_path = os.path.join(output_dir, 'ijkl_action.npy')
    
    np.save(wasd_output_path, wasd_actions)
    np.save(rotation_output_path, rotation_actions)

    print(wasd_actions)
    print(rotation_actions)

    print(f"WASD actions shape: {wasd_actions.shape}")
    print(f"Rotation actions shape: {rotation_actions.shape}")
    print(f"Saved WASD actions to: {wasd_output_path}")
    print(f"Saved rotation actions to: {rotation_output_path}")
    