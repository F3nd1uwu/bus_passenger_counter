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

Опциональный веб-кабинет использует отдельно закреплённые зависимости из
`requirements-web.txt`, включая Django, Gunicorn и WhiteNoise. Их лицензии
сохраняются в установленных дистрибутивах; веб-образ не содержит моделей распознавания.

## OCR часов камеры

Локальная модель PP-OCRv4 получена из опубликованного дистрибутива
[RapidOCR 1.4.4](https://pypi.org/project/rapidocr-onnxruntime/1.4.4/).
Хеш дистрибутива, имя файла и хеш ONNX: [models/camera_clock.json](models/camera_clock.json).
Сохранён [Apache-2.0 LICENSE](third_party/CAMERA_CLOCK_LICENSE.txt) проекта
[RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR).
Используется только распознавание фиксированного участка часов через ONNX Runtime;
пакет RapidOCR не устанавливается, кадры во внешние сервисы не передаются.
