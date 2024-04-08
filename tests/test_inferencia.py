import unittest
from unittest.mock import Mock, patch
import sys
import numpy as np


# Agregar la ruta del proyecto al path del sistema
sys.path.append('/Users/ignaciocarrenoromero/proyecto_escalada')

# Importar la clase Inferencia desde el módulo app.utils.inferencia
from app.utils.inferencia import Inferencia
class TestInferencia(unittest.TestCase):
    """
    Clase para realizar pruebas unitarias a la clase Inferencia.
    """

    @patch('app.utils.inferencia.AutoModelForObjectDetection')
    @patch('app.utils.inferencia.AutoImageProcessor')
    def setUp(self, mock_processor, mock_model):
        """
        Configuración inicial para las pruebas.
        Crea una instancia de la clase Inferencia con un modelo y procesador simulados.
        """
        self.inferencia = Inferencia(mock_model, mock_processor)

    def test_inferencia_imagen(self):
        """
        Prueba el método inferencia_imagen de la clase Inferencia.
        """
        with patch('app.utils.inferencia.Image.open') as mock_open:
            mock_open.return_value.size = [640, 480]
            detecciones = self.inferencia.inferencia_imagen('ruta/a/imagen.jpg')
            self.assertIsInstance(detecciones, list)

    def test_inferencia_imagen_URL(self):
        """
        Prueba el método inferencia_imagen_URL de la clase Inferencia.
        """
        with patch('app.utils.inferencia.requests.get') as mock_get:
            mock_get.return_value.raw = Mock()
            with patch('app.utils.inferencia.Image.open') as mock_open:
                mock_open.return_value.size = [640, 480]
                detecciones = self.inferencia.inferencia_imagen_URL('http://ruta/a/imagen.jpg')
                self.assertIsInstance(detecciones, list)

    @patch('app.utils.inferencia.VideoStream')
    @patch('app.utils.inferencia.cv2')
    def test_inferencia_video(self, mock_cv2, mock_vs):
        """
        Prueba el método inferencia_video de la clase Inferencia.
        """
        mock_stream = Mock()
        mock_vs.return_value = mock_stream
        mock_stream.read.side_effect = [np.ones((480, 640, 3), dtype=np.uint8), None]  # Simula un frame de video y luego None para detener el bucle
        mock_cv2.COLOR_BGR2RGB = 'COLOR_BGR2RGB'
        mock_cv2.waitKey.return_value = ord('q')  # Simula la pulsación de la tecla 'q' para detener el bucle

        self.inferencia.inferencia_video('ruta/a/video.mp4')

        # Verifica que se llamaron los métodos esperados
        mock_vs.assert_called_once_with(src='ruta/a/video.mp4')
        mock_stream.start.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_cv2.destroyAllWindows.assert_called_once()

    @patch('app.utils.inferencia.VideoStream')
    @patch('app.utils.inferencia.cv2')
    def test_inferencia_video_directo(self, mock_cv2, mock_vs):
        """
        Prueba el método inferencia_video_directo de la clase Inferencia.
        """
        mock_stream = Mock()
        mock_vs.return_value = mock_stream
        mock_stream.read.side_effect = [np.ones((480, 640, 3), dtype=np.uint8), None]  # Simula un frame de video y luego None para detener el bucle
        mock_cv2.COLOR_BGR2RGB = 'COLOR_BGR2RGB'
        mock_cv2.waitKey.return_value = ord('q')  # Simula la pulsación de la tecla 'q' para detener el bucle

        self.inferencia.inferencia_video_directo()

        # Verifica que se llamaron los métodos esperados
        mock_vs.assert_called_once_with(src=0, resolution=(640, 480))
        mock_stream.start.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_cv2.destroyAllWindows.assert_called_once()
if __name__ == '__main__':
    unittest.main()