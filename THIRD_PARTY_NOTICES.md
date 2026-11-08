# Сторонние компоненты

## Ultralytics

Детекция и сопровождение используют пакет Ultralytics. Включённые базовый ONNX
YOLO26n и его дообученный вариант происходят из этой экосистемы. Версия пакета
закреплена в `requirements.txt`.

Текст лицензии из установленного дистрибутива Ultralytics сохранён без изменений в
[third_party/ULTRALYTICS_LICENSE.txt](third_party/ULTRALYTICS_LICENSE.txt).
Исходный проект: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics),
[лицензия проекта](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## OSNet / Torchreid

Архитектура OSNet заимствована из Torchreid с сохранением авторства и MIT-лицензии:
[tools/vendor/OSNET_LICENSE.txt](tools/vendor/OSNET_LICENSE.txt).
Исходный проект: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).
Источник конкретных весов, ревизия и контрольные суммы указаны в
[models/person_reid.json](models/person_reid.json).

Остальные библиотеки устанавливаются как зависимости и сохраняют собственные
лицензии. Этот файл фиксирует происхождение сторонних компонентов и не назначает
новую лицензию авторскому коду проекта.
