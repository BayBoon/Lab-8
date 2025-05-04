import cv2
import numpy as np
import math
import os


def main():
    # === Настройка путей ===
    IMG_PATH = os.path.join("images", "variant-7.jpg")  # Путь к изображению метки
    FLY_PATH = "fly64.png"  # Путь к изображению мухи

    # Чтение и проверка изображения метки
    marker_image = cv2.imread(IMG_PATH)
    if marker_image is None:
        raise FileNotFoundError("Файл variant-7.jpg не найден в папке images")

    # Отражаем изображение по горизонтали и вертикали (вариант 7)
    processed_marker = cv2.flip(marker_image, -1)
    cv2.imwrite("processed_marker.jpg", processed_marker)

    # Загрузка изображения мухи
    fly_img = cv2.imread(FLY_PATH, cv2.IMREAD_UNCHANGED)  # Загружаем с альфа-каналом
    if fly_img is None:
        raise FileNotFoundError("Файл fly64.png не найден")

    has_alpha = fly_img.shape[2] == 4  # Проверяем наличие альфа-канала

    # Настройка видеозахвата
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видеокамеру")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем в HSV для цветового детектирования
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Диапазоны HSV для красного цвета (метка)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Создаем маски для красного цвета
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Находим контуры на маске
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # Выбираем наибольший контур
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            
            if moments['m00'] != 0:
                # Вычисляем центр метки
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])

                # Рисуем центр метки
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # Вычисления расстояния до центра кадра
                frame_height, frame_width = frame.shape[:2]
                frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
                distance = math.hypot(center_x - frame_center_x, center_y - frame_center_y)

                # Вывод расстояния
                cv2.putText(
                    frame,
                    f'{int(distance)} px',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                #Наложение изображения мухи
                fly_height, fly_width = fly_img.shape[:2]
                fly_top_left_x = center_x - fly_width // 2
                fly_top_left_y = center_y - fly_height // 2

                # Проверяем, чтобы муха не выходила за границы кадра
                if (0 <= fly_top_left_x < frame_width - fly_width and
                        0 <= fly_top_left_y < frame_height - fly_height):
                    
                    if has_alpha:
                        # Альфа-смешение для прозрачности
                        roi = frame[
                            fly_top_left_y:fly_top_left_y + fly_height,
                            fly_top_left_x:fly_top_left_x + fly_width
                        ]
                        alpha = fly_img[:, :, 3] / 255.0
                        
                        for channel in range(3):
                            roi[:, :, channel] = (
                                (1 - alpha) * roi[:, :, channel] + 
                                alpha * fly_img[:, :, channel]
                            )
                    else:
                        # Простое наложение без прозрачности
                        frame[
                            fly_top_left_y:fly_top_left_y + fly_height,
                            fly_top_left_x:fly_top_left_x + fly_width
                        ] = fly_img

        # результат
        cv2.imshow("Отслеживание метки с мухой", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Выход по ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()