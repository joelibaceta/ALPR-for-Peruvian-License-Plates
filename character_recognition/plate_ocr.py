import re
import subprocess
import json
from pathlib import Path
import tempfile
import cv2

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None  # Si no está disponible, se usará el modo CLI

class PlateOCR:
    def __init__(self, model_path=None, lang='en', use_gpu=False, use_cli=False):
        self.use_cli = use_cli or PaddleOCR is None
        self.lang = lang
        self.model_path = model_path

        if not self.use_cli:
            self.ocr_model = PaddleOCR(use_angle_cls=True, lang=lang)

    def recognize(self, plate_image):
        if self.use_cli:
            return self._recognize_cli(plate_image)
        else:
            return self._recognize_embedded(plate_image)

    def recognize_valid(self, plate_image):
        texto = self.recognize(plate_image)
        return texto if self._es_placa_valida(texto) else ''

    def _recognize_embedded(self, plate_image):
        result = self.ocr_model.ocr(plate_image, cls=True)
        if result and result[0]:
            textos = [entry[1][0] for entry in result[0] if len(entry[1][0]) >= 5]
            if textos:
                return textos[0].strip()
        return ''

    def _recognize_cli(self, plate_image):
        # Si es una ruta, usar directamente
        if isinstance(plate_image, (str, Path)):
            image_path = Path(plate_image)
            if not image_path.exists():
                return ''
        else:
            # Guardar imagen temporal si es un ndarray
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, plate_image)
            image_path = Path(tmp.name)

        command = [
            "paddleocr", "ocr",
            "--image_dir", str(image_path),
            "--use_angle_cls", "true",
            "--lang", self.lang
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = json.loads(result.stdout)
            if isinstance(output, list) and output and isinstance(output[0], list):
                textos = [entry[1][0] for entry in output[0] if len(entry[1][0]) >= 5]
                return textos[0].strip() if textos else ''
        except Exception as e:
            print(f"❌ Error ejecutando PaddleOCR CLI: {e}")
        finally:
            # Limpieza del archivo temporal (si lo generaste)
            if 'tmp' in locals():
                image_path.unlink(missing_ok=True)

        return ''

    def _es_placa_valida(self, texto):
        texto = texto.upper().replace(" ", "").replace("\n", "")
        patrones = [
            r"^[A-Z]{3}-?\d{3}$",        # ABC-123
            r"^[A-Z][0-9][A-Z]-?\d{3}$", # A1B-234
            r"^[A-Z][0-9]{2}-?\d{3}$",   # A12-345
            r"^[A-Z]{2}-?\d{4}$",        # AB-1234
            r"^\d{4}-?[A-Z]{2}$",        # 1234-AB
            r"^\d{2}[A-Z]-?\d{3}$",      # 68D-447
            r"^[A-Z][0-9]-?\d{3}$",      # B3-432
            r"^[A-Z]{2}[0-9]-?\d{3}$",   # BF0-783
            r"^[A-Z]{2}-?\d{3}$",        # AN-394
        ]
        return any(re.fullmatch(p, texto) for p in patrones)