import insightface
import os
import cv2

from datetime import datetime
import pickle

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from config import Config

conf = Config()


def extract_embedding(
    probe_name: str,
    folder_path: str = conf.GALLERY_DIR,
    save_path: str = conf.OUTPUT_DIR,
    verbose: bool = True,
):

    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    try:
        process_count = 0
        detected = 0
        not_detected = 0
        error_count = 0
        embeddings_dict = {}
        error_messages = []

        for filename in os.listdir(folder_path):

            if process_count % 2000 == 0:
                print(f"{process_count} images embedded")

            if filename.endswith(".jpg") or filename.endswith(
                ".png"
            ):  # Adjust based on your image format
                filepath = os.path.join(folder_path, filename)
                key_name = os.path.join(probe_name, filename)

                img = cv2.imread(filepath)  # Read the image using OpenCV

                try:
                    faces = app.get(img)
                    if faces:
                        embeddings_dict[key_name] = faces[0].embedding
                        detected += 1
                        # print(f"{len(faces)} face detected with max confidence: {faces[0].det_score}")
                    else:
                        not_detected += 1
                        # print("No face detected")
                except Exception as e:
                    msg = f"ERROR: while processing image with shape {img.shape} at\n{filepath}: {e}\n"
                    print(msg)
                    error_messages.append(msg)
                    error_count += 1

                process_count += 1

        print(f"Process Complete : {process_count} processed")
        print(
            f"detected:{detected}, not_detected:{not_detected}, error_count:{error_count}"
        )

        misc = {
            "process_count": process_count,
            "detected": detected,
            "not_detected": not_detected,
            "error_count": error_count,
            "error_messages": error_messages,
        }

        return embeddings_dict, misc

    except Exception as e:
        raise e


def save_embedding(
    file_name: str,
    embeddings_dict: dict,
    misc: dict,
    output_path: str = conf.OUTPUT_DIR,
):
    pack = {}
    pack["embeddings_dict"] = embeddings_dict
    pack["misc"] = misc

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_path = os.path.join(output_path, f"{file_name}_{current_date}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(pack, f)

    print(f"Embeddings saved to:\n{file_path}")
