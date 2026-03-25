import cv2
import os

input_dir = 'data/cyberpunk_temporal'
output_file = 'data/cyberpunk_temporal_test.mp4'

if not os.path.exists(input_dir):
    print(f"Error: {input_dir} not found!")
    exit(1)

# Extract all frames and naturally sort their index 
frames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
# The regex parses the raw integer out of 'frame_X.png'
frames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

if not frames:
    print("No PNG frames found in directory.")
    exit(1)

# Read first frame to lock dimensions into the video wrapper
first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
h, w, layers = first_frame.shape

print(f"===========================================================")
print(f"| MYELIN-SR | Compiling Video Sequence                     ")
print(f"===========================================================")
print(f"Ingesting: {len(frames)} frames")
print(f"Resolution: {w}x{h} (1440p Target)")
print(f"Framerate: 60 FPS")

# Codec execution (MP4 Video fallback compatible with basic Windows players)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_file, fourcc, 60.0, (w, h))

for i, frame_name in enumerate(frames):
    img_path = os.path.join(input_dir, frame_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Corruption on {img_path}. Skipping frame.")
        continue
        
    out.write(img)
    if (i + 1) % 45 == 0:
        print(f"[Encoder] Stitched {i + 1}/{len(frames)} frames...")

out.release()
print(f"\n[Encoder] Video generation finalized perfectly!")
print(f"Test Evidence available at: {output_file}\n")
