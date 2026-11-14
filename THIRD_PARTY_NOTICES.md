# Сторонние компоненты

Детекция людей и жилетки использует Ultralytics. Версии библиотек закреплены
в `requirements.txt`. Текст лицензии установленного дистрибутива:
[ULTRALYTICS_LICENSE.txt](third_party/ULTRALYTICS_LICENSE.txt).
Исходный проект: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

OSNet происходит из [Torchreid](https://github.com/KaiyangZhou/deep-person-reid).
Сохранена [MIT-лицензия OSNet](third_party/OSNET_LICENSE.txt).
Источник весов и хеши указаны в [метаданных ReID](models/person_reid.json).

Метаданные детекторов находятся рядом с файлами ONNX. Остальные зависимости
сохраняют свои лицензии. Этот файл фиксирует происхождение сторонних компонентов,
а не назначает лицензию авторскому коду проекта. Кадры пассажиров и тренировочные
датасеты в поставку не входят.
