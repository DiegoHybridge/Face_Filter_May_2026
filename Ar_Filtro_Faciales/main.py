import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection
from filters.filtro_nariz import FiltroNariz


def main():
    cameras = CameraManager.list_available(5)
    if not cameras:
        print("No se detectaron cámaras activas.")
        return

    print("Cámaras detectadas:")
    print("  ", " ".join(str(cam) for cam in cameras))

    choice = input("Elige el número de la cámara (enter = primera): ").strip()
    selected = cameras[0] if choice == "" else int(choice)

    if selected not in cameras:
        print(f"Cámara {selected} no está en la lista.")
        return

    camera = CameraManager(selected)
    detector = FaceDetection()
    filter_nose = FiltroNariz(detector)

    print(f"Cámara {selected} seleccionada y abierta.")
    print("Presiona 'q' para cerrar la vista previa.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("No se pudo leer el frame de la cámara.")
                break

            faces = detector.detect_face(frame)
            output = filter_nose.apply(frame, faces)
            cv2.imshow("Filtro nariz", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()