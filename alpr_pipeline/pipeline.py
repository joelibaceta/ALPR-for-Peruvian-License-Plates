import cv2
from vehicle_detection.detector import VehicleDetector
from plate_segmentation.plate_segmentation import PlateSegmenter
from character_recognition.plate_ocr import PlateOCR
import os

os.makedirs("debug_plates", exist_ok=True)  

class ALPRPipeline:
    def __init__(self,
                 vehicle_model_path,
                 vehicle_cfg_path,
                 plate_model_path,
                 plate_cfg_path,
                 vehicle_nc=3,
                 plate_nc=1,
                 ocr_model_path=None):

        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path,
            config_path=vehicle_cfg_path,
            nc=vehicle_nc
        )

        self.plate_segmenter = PlateSegmenter(
            model_path=plate_model_path,
            config_path=plate_cfg_path,
            nc=plate_nc
        )

        self.ocr_model = PlateOCR(model_path=ocr_model_path)

    def process_image(self, img, draw=True):
        original = img.copy()
        vehicle_results = self.vehicle_detector.detect(img)
        placas_validas = []

        for vehicle in vehicle_results:
            vx1, vy1, vx2, vy2 = vehicle['box']
            vehicle_crop = img[vy1:vy2, vx1:vx2]
            print(f"crop_vehicle: {vx1}, {vy1}, {vx2}, {vy2}")
            plate_results = self.plate_segmenter.segment(vehicle_crop)

            cv2.rectangle(original, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)

            # Dentro de process_image
            for plate in plate_results:
                px1, py1, px2, py2 = plate['box']
                print(f"crop_plate: {px1}, {py1}, {px2}, {py2}")
                abs_box = [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                abs_box = [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]


                color = (0, 255, 0)
                cv2.rectangle(original, (abs_box[0], abs_box[1]), (abs_box[2], abs_box[3]), color, 2)


                # OCR
                texto_raw = self.ocr_model.recognize(plate_crop)
                print("🔤 Texto OCR bruto:", texto_raw)
                texto_valido = texto_raw if self.ocr_model._es_placa_valida(texto_raw) else ''

                if texto_valido:
                    placas_validas.append({
                        'plate_text': texto_valido,
                        'plate_box': abs_box,
                        'vehicle_box': [vx1, vy1, vx2, vy2]
                    })

                    if draw:
                        cv2.rectangle(original, (abs_box[0], abs_box[1]), (abs_box[2], abs_box[3]), (0, 255, 0), 2)
                        cv2.putText(original, texto_valido, (abs_box[0], abs_box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return original if draw else placas_validas

    def process_video(self, video_path, fps=5):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                annotated = self.process_image(frame, draw=True)
                cv2.imshow("ALPR Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()