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
        """
        Initialize the ALPR (Automatic License Plate Recognition) pipeline.

        This constructor sets up the complete pipeline for license plate recognition,
        consisting of vehicle detection, plate segmentation, and optical character
        recognition (OCR) for Peruvian license plates.

        Parameters
        ----------
        vehicle_model_path : str
            Path to the pre-trained vehicle detection model.
        vehicle_cfg_path : str
            Path to the configuration file for the vehicle detection model.
        plate_model_path : str
            Path to the pre-trained plate segmentation model.
        plate_cfg_path : str
            Path to the configuration file for the plate segmentation model.
        vehicle_nc : int, optional
            Number of classes for vehicle detection, default is 3.
        plate_nc : int, optional
            Number of classes for plate detection, default is 1.
        ocr_model_path : str, optional
            Path to the pre-trained OCR model. If None, uses a default model.

        Notes
        -----
        The pipeline consists of three main components:
        1. Vehicle detection using YOLOv5
        2. License plate segmentation
        3. Optical character recognition for extracting text from license plates
        """

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
        """
        Process an image to detect vehicles, segment license plates, and perform OCR on plate regions.
        
        This method performs the complete ALPR (Automatic License Plate Recognition) pipeline:
        1. Detects vehicles in the input image
        2. For each vehicle, segments potential license plate regions
        3. Applies OCR to recognize text on plate regions
        4. Validates the recognized text according to license plate format rules
        5. Optionally draws bounding boxes around vehicles and license plates with recognized text
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image to process (BGR format)
        draw : bool, default=True
            Whether to draw detection results on the image
            - If True, returns annotated image with bounding boxes and plate text
            - If False, returns a list of detected license plates with their coordinates
        
        Returns:
        --------
        numpy.ndarray or list:
            If draw=True: Returns the original image with annotations
            If draw=False: Returns a list of dictionaries containing:
                {
                    'plate_text': Recognized and validated license plate text,
                    'plate_box': Absolute coordinates [x1, y1, x2, y2] of the plate in the original image,
                    'vehicle_box': Coordinates [x1, y1, x2, y2] of the vehicle in the original image
                }
        """
        original = img.copy()
        vehicle_results = self.vehicle_detector.detect(img)
        placas_validas = []

        for vehicle in vehicle_results:
            vx1, vy1, vx2, vy2 = vehicle['box']
            vehicle_crop = img[vy1:vy2, vx1:vx2] 
            plate_results = self.plate_segmenter.segment(vehicle_crop)

            for plate in plate_results:
                cv2.rectangle(original, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)

                px1, py1, px2, py2 = plate['box'] 
                abs_box = [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                abs_box = [vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]

                color = (0, 255, 0)
                cv2.rectangle(original, (abs_box[0], abs_box[1]), (abs_box[2], abs_box[3]), color, 2)

                # OCR
                texto_raw = self.ocr_model.recognize(plate_crop) 
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
        """
        Process a video file by applying the ALPR pipeline on frames at a specified frequency.
        
        This method opens a video file, samples frames at the requested fps rate, applies
        the license plate recognition process to each sampled frame, and displays the 
        annotated results in a window. The processing continues until the end of the video
        or until the user presses 'q'.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file to process.
        fps : int, optional
            The target processing rate in frames per second. Default is 5.
            This controls how many frames from the original video will be processed.
            Lower values improve performance but might miss fast-moving plates.
        
        Notes:
        ------
        - The actual processing interval is calculated based on the video's native fps.
        - Press 'q' to exit processing before the end of the video.
        - This method calls process_image() internally for each selected frame.
        """
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