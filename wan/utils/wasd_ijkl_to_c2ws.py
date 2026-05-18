import torch
import numpy as np

_ALLOWED_KEYS = frozenset("wasdijkl")


def normalize_action_string(action_string: str) -> str:
    """Normalize commas and remove all whitespace from user input."""
    if not action_string:
        return ""
    normalized = action_string.replace("，", ",")
    return "".join(normalized.split())


def parse_action_string_segments(action_string: str):
    """
    Parse a compact DSL like ``w-3,iw-1,none-5,ijd-5``.

    Each segment is ``<keys>-<duration>`` where the last ``-`` separates keys from the
    integer frame count. ``none`` means no keys for that run.

    Returns:
        tuple: (segments, total_frames) where segments is a list of
        (frozenset of key chars, num_frames).
    """
    s = normalize_action_string(action_string)
    if not s:
        raise ValueError("action_string is empty")
    raw_parts = s.split(",")
    parts = [p for p in raw_parts if p]
    if len(parts) != len(raw_parts):
        raise ValueError("action_string has an empty segment (check commas)")
    segments = []
    total = 0
    for part in parts:
        if "-" not in part:
            raise ValueError(
                f"Invalid segment {part!r}: expected form <keys>-<duration>, e.g. w-3 or none-5"
            )
        keys_part, dur_str = part.rsplit("-", 1)
        if not dur_str.isdigit():
            raise ValueError(
                f"Invalid duration in segment {part!r}: {dur_str!r} is not a positive integer"
            )
        n = int(dur_str)
        if n <= 0:
            raise ValueError(f"Duration must be positive in segment {part!r}")
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys = frozenset()
        else:
            bad = [c for c in keys_lower if c not in _ALLOWED_KEYS]
            if bad:
                raise ValueError(
                    f"Invalid key character(s) {bad!r} in segment {part!r}; "
                    "allowed: w,a,s,d,i,j,k,l"
                )
            keys = frozenset(keys_lower)
        segments.append((keys, n))
        total += n
    return segments, total


def segments_to_wasd_ijkl(segments):
    """Build (F,4) WASD and IJKL float arrays (0/1) from parsed segments."""
    total = sum(n for _, n in segments)
    wasd = np.zeros((total, 4), dtype=np.float32)
    ijkl = np.zeros((total, 4), dtype=np.float32)
    wasd_idx = {"w": 0, "a": 1, "s": 2, "d": 3}
    ijkl_idx = {"i": 0, "j": 1, "k": 2, "l": 3}
    t = 0
    for keys, n in segments:
        for _ in range(n):
            for c in keys:
                if c in wasd_idx:
                    wasd[t, wasd_idx[c]] = 1.0
                else:
                    ijkl[t, ijkl_idx[c]] = 1.0
            t += 1
    assert t == total
    return wasd, ijkl


def action_string_to_wasd_ijkl(action_string: str):
    """
    Convert ``action_string`` to the same layout as ``wasd_action.npy`` / ``ijkl_action.npy``.

    Returns:
        tuple: (wasd_action, ijkl_action, total_frames)
    """
    segments, total = parse_action_string_segments(action_string)
    wasd, ijkl = segments_to_wasd_ijkl(segments)
    return wasd, ijkl, total


def infer_frame_num_from_action_string(action_string: str) -> int:
    _, total = parse_action_string_segments(action_string)
    return total


def pad_frame_num_to_4n_plus_1(frame_num: int) -> int:
    """
    Return the smallest value >= frame_num that satisfies F = 4n + 1.
    """
    if frame_num <= 0:
        raise ValueError("frame_num must be positive")
    remainder = (frame_num - 1) % 4
    if remainder == 0:
        return frame_num
    return frame_num + (4 - remainder)


def wasd_array_to_frame_keys(wasd_array, ijkl_array=None):
    """
    Convert numpy arrays (representing WASD and IJKL key states, ranged 0-1) into a list of lists of pressed keys for each frame.

    Args:
        wasd_array (np.ndarray): A 2D array of shape (num_frames, 4) where each column corresponds to 'w', 'a', 's', 'd' keys respectively, and values indicate the intensity of the key press (0 to 1).
        ijkl_array (np.ndarray, optional): A 2D array of shape (num_frames, 4) where each column corresponds to 'i', 'j', 'k', 'l' keys respectively.
    Returns:
        List[List[str]]: A list where each element is a list of keys pressed in that frame. For example, if the first frame has 'w' and 'j' pressed, it would be ['w', 'j']. If no keys are pressed, it would be an empty list [].
    """
    wasd_mapping = ['w', 'a', 's', 'd']
    ijkl_mapping = ['i', 'j', 'k', 'l']
    frame_keys = []
    
    for idx, frame in enumerate(wasd_array):
        pressed_keys = [wasd_mapping[i] for i in range(4) if frame[i] > 0.5]
        
        if ijkl_array is not None:
            ijkl_frame = ijkl_array[idx]
            pressed_keys += [ijkl_mapping[i] for i in range(4) if ijkl_frame[i] > 0.5]
        
        frame_keys.append(pressed_keys)
    return frame_keys


def get_rotation_matrix(axis, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)

def generate_and_save_trajectory(arrow_actions):
    move_speed = 0.05
    rotate_speed_rad = np.deg2rad(2.0)
    
    current_c2w = np.eye(4)
    current_pitch = 0.0
    pitch_limit = np.deg2rad(85)
    
    all_matrices = []
    all_matrices.append(current_c2w)

    total_frames = len(arrow_actions)

    for f in range(total_frames):
        frame_keys = arrow_actions[f]

        R = current_c2w[:3, :3]
        T = current_c2w[:3, 3]

        pitch_delta = 0.0
        if 'i' in frame_keys: pitch_delta += rotate_speed_rad
        if 'k' in frame_keys: pitch_delta -= rotate_speed_rad
        
        new_pitch = current_pitch + pitch_delta
        if -pitch_limit <= new_pitch <= pitch_limit:
            current_pitch = new_pitch
        else:
            pitch_delta = 0.0
        
        R_pitch = get_rotation_matrix('x', pitch_delta)

        yaw_delta = 0.0
        if 'j' in frame_keys: yaw_delta -= rotate_speed_rad
        if 'l' in frame_keys: yaw_delta += rotate_speed_rad
        R_yaw = get_rotation_matrix('y', yaw_delta)

        R_new = R_yaw @ R @ R_pitch

        vec_right = R_new[:, 0]
        vec_forward = R_new[:, 2]

        forward_flat = np.array([vec_forward[0], 0, vec_forward[2]])
        right_flat   = np.array([vec_right[0],   0, vec_right[2]])

        f_norm = np.linalg.norm(forward_flat)
        r_norm = np.linalg.norm(right_flat)
        forward_flat = forward_flat / (f_norm + 1e-6) if f_norm > 0 else forward_flat
        right_flat   = right_flat / (r_norm + 1e-6) if r_norm > 0 else right_flat

        move_vec = np.zeros(3)
        if 'w' in frame_keys: move_vec += forward_flat * move_speed
        if 's' in frame_keys: move_vec -= forward_flat * move_speed
        if 'd' in frame_keys: move_vec += right_flat * move_speed
        if 'a' in frame_keys: move_vec -= right_flat * move_speed

        T_new = T + move_vec
        current_c2w = np.eye(4)
        current_c2w[:3, :3] = R_new
        current_c2w[:3, 3] = T_new
        
        all_matrices.append(current_c2w)

    return all_matrices


if __name__ == "__main__":
    keyboard_actions=[["d", "w", "a"], [], [], [], ["s"], [], ["s"], ["s"], ["s"], ["s"]]
    c2ws = generate_and_save_trajectory(keyboard_actions)

    wasd_array = np.array([
        [1, 0, 0, 0],  # w
        [0, 0, 1, 0],  # s
    ])
    
    ijkl_array = np.array([
        [0, 1, 0, 0],  # j
        [1, 0, 0, 0],  # i
    ])

    frame_keys = wasd_array_to_frame_keys(wasd_array, ijkl_array)
    print(frame_keys)  # [['w', 'j'], ['s', 'i']]

