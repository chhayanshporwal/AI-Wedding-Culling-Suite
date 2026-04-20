import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw

# Add parent dir to path so we can import pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import run_filtering

def create_dummy_image(path):
    # Create a 640x480 RGB image with random noise and a rectangle (face-like)
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Hello World", fill=(255, 255, 0))
    d.rectangle([200, 150, 440, 390], outline="red", width=3)
    img.save(path)
    print(f"Created dummy image at {path}")

def main():
    base_dir = os.path.dirname(__file__)
    test_in = os.path.join(base_dir, "test_input_temp")
    test_out = os.path.join(base_dir, "test_output_temp")
    
    # Cleanup previous runs
    if os.path.exists(test_in): shutil.rmtree(test_in)
    if os.path.exists(test_out): shutil.rmtree(test_out)
    
    os.makedirs(test_in)
    os.makedirs(test_out)

    try:
        # Create 3 dummy images
        create_dummy_image(os.path.join(test_in, "test1.jpg"))
        create_dummy_image(os.path.join(test_in, "test2.jpg"))
        create_dummy_image(os.path.join(test_in, "test3.jpg"))

        print("Running pipeline...")
        out_path = run_filtering(input_folder=test_in, output_base=test_out, max_workers=2)
        
        if out_path and os.path.exists(out_path):
            print(f"✅ Success! Output generated at: {out_path}")
            csv_path = os.path.join(out_path, "log.csv")
            if os.path.exists(csv_path):
                 print(f"✅ Log file found at: {csv_path}")
            else:
                 print("❌ Log file missing!")
        else:
            print("❌ Pipeline failed to generate output.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup input
        if os.path.exists(test_in): shutil.rmtree(test_in)
        # We assume user might want to see output, so we won't delete test_out automatically
        # or maybe we should to keep it clean? 
        # Let's keep it for inspection if user wants, but print where it is.
        pass

if __name__ == "__main__":
    main()
