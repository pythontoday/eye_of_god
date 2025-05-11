from ultralytics import YOLO
from collections import Counter
from pathlib import Path


def detect_objects(model: YOLO, image_path: str) -> None:
    """
    Выполняет детекцию объектов на изображении и сохраняет результат в указанный файл.
    Также выводит в терминал количество найденных объектов по категориям.

    :param model: загруженная модель YOLO
    :param image_path: путь к изображению, на котором нужно произвести детекцию.
    """

    print("[INFO] Запускаем распознавание объектов...")
    results = model(image_path, verbose=False)[0]

    # print(results)

    if results.names and results.boxes is not None:
        # Получаем список идентификаторов классов объектов
        labels = results.boxes.cls.tolist()
        # Преобразуем ID в названия классов
        label_names = [results.names[int(cls)] for cls in labels]
        # Считаем количество каждого уникального объекта
        counts = Counter(label_names)

        print("[INFO] Обнаруженные объекты:")
        for label, count in counts.items():
            print(f'[+] {label}: {count}')
    else:
        print('[!] Объекты не обнаружены!')

    save_path = results.save(filename=f"RESULT_{Path(image_path).stem}.png")
    print(f"[+] Результат сохранен в файл: {save_path}")

def main():
    model = YOLO('yolo11n.pt')

    detect_objects(model, 'img_path')


if __name__ == '__main__':
    main()
