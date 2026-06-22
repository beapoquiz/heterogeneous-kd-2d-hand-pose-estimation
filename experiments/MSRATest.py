import numpy as np
from mmpose.codecs import MSRAHeatmap

# 1. Initialize the Bridge
codec = MSRAHeatmap(
    input_size=(256, 256), 
    heatmap_size=(64, 64), 
    sigma=2, 
    unbiased=True
)

# 2. CREATE THE DATA CORRECTLY
# We use (21, 64, 64) -> This is exactly 3 values (K, H, W)
fake_heatmap = np.random.randn(21, 64, 64).astype(np.float32)

print("--- STARTING DECODER TEST ---")

try:
    # 3. Translate the "Glow" into "Numbers"
    # In MMPose v1.x, decode returns a dictionary or a tuple
    result = codec.decode(fake_heatmap)
    
    print("SUCCESS: The Bridge is working!")
    
    # Let's see what the coordinates look like
    if isinstance(result, dict):
        coords = result['keypoints']
    else:
        # If it's a tuple, the coords are usually the first item
        coords = result[0]
        
    print(f"Coordinates Shape: {coords.shape}") # Should be (21, 2)
    print(f"Sample (Wrist Joint): {coords[0]}")
    print("-----------------------------")

except Exception as e:
    print(f"ERROR: {e}")