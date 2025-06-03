class ALPRPipeline:
    def __init__(self, detector, segmenter, ocr):
        self.detector = detector
        self.segmenter = segmenter
        self.ocr = ocr

    def run(self, image):
        vehicle_boxes = self.detector.detect(image)
        results = []
        for box in vehicle_boxes:
            vehicle_crop = self._crop(image, box)
            plate_box = self.segmenter.segment(vehicle_crop)
            plate_crop = self._crop(vehicle_crop, plate_box)
            text = self.ocr.recognize(plate_crop)
            results.append({'box': box, 'plate': plate_box, 'text': text})
        return results

    def _crop(self, image, box):
        # Recortar la imagen según el bounding box
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]