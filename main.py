import argparse
import cv2
from alpr_pipeline.pipeline import ALPRPipeline


class ALPRApp:
    def __init__(self):
        self.pipeline = ALPRPipeline(
            vehicle_model_path='vehicle_detection/weights/yolov7_vehicle_weights.pt',
            vehicle_cfg_path='yolo_core/cfg/yolov7_vehicles.yaml',
            plate_model_path='plate_segmentation/weights/yolov7_plate_weights.pt',
            plate_cfg_path='yolo_core/cfg/yolov7_plates.yaml',
            ocr_model_path='character_recognition/weights/plate_ocr_model.pt'  # si aplica
        )

    def run_on_image(self, image_path, show=True, save_path=None):
        img = cv2.imread(image_path)
        result = self.pipeline.process_image(img, draw=True)
        if show:
            cv2.imshow("ALPR Resultado", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path:
            cv2.imwrite(save_path, result)

    def run_on_video(self, video_path):
        self.pipeline.process_video(video_path, fps=5)


# ==== CLI ====
def main():
    parser = argparse.ArgumentParser(description="ALPR para placas peruanas")
    parser.add_argument("--image", type=str, help="Ruta a una imagen para procesar")
    parser.add_argument("--video", type=str, help="Ruta a un video para procesar")
    parser.add_argument("--save", type=str, help="Ruta para guardar el resultado (solo imagen)")
    parser.add_argument("--no-display", action="store_true", help="No mostrar resultados visuales")

    args = parser.parse_args()

    app = ALPRApp()

    if args.image:
        app.run_on_image(
            image_path=args.image,
            show=not args.no_display,
            save_path=args.save
        )
    elif args.video:
        app.run_on_video(args.video)
    else:
        print("❌ Debes especificar una imagen con --image o un video con --video")


if __name__ == "__main__":
    main()