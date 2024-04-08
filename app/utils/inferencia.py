
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from imutils.video import VideoStream
import torch
import time
from PIL import Image
import cv2
import requests

class Inferencia:
    

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor


    def inferencia_imagen(self, img_path):
        img = Image.open(img_path)
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detecciones = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            etiqueta = self.model.config.id2label.get(
                label.item(), "Desconocido"
            )
            detecciones.append((etiqueta, score.item()))

        return detecciones

    def inferencia_imagen_URL(self, url):
        img = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detecciones = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            etiqueta = self.model.config.id2label.get(label.item(), "Desconocido")
            detecciones.append((etiqueta, score.item()))

        return detecciones

    def inferencia_video(self, video_path):
        vs = VideoStream(src=video_path).start()
        time.sleep(2.0)
        while True:
            frame = vs.read()
            if frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self.processor(pil_img, return_tensors="pt")
            outputs = self.model(**inputs)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        vs.stop()
        cv2.destroyAllWindows()

    def inferencia_video_directo(self, fuente=0):
        vs = VideoStream(src=fuente, resolution=(640, 480)).start()
        time.sleep(2.0)
        while True:
            frame = vs.read()
            print(type(frame), frame.shape)

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.9
            )[0]
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                box = [int(i) for i in box.tolist()]
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}"
                )
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    self.model.config.id2label[label.item()],
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        vs.stop()
        cv2.destroyAllWindows()
