import os,shutil
from ultralytics import YOLO
import cv2

MODEL_PATH = "yolov8n.pt"
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output"

def detect_object():
    model = YOLO(MODEL_PATH)

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
        os.mkdir(OUTPUT_FOLDER)

    for image in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER,image)
        output_path = os.path.join(OUTPUT_FOLDER,image)
          
        if not input_path.lower().endswith(('.png','jpg','jpeg')):
            continue
        results = model.predict(input_path, conf=0.5, save=False)

        for result in results:
            annotated_image = result.plot(result)
            cv2.imwrite(output_path,annotated_image)
            print(f"Hasil deteksi disimpan ke {image}")


    pass
if __name__ == __main__:
    detect_object()

