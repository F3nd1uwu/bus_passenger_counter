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

## Экспериментальный фильтр детей в полной ветке

TinyCLIP: [Microsoft Cream / TinyCLIP](https://github.com/microsoft/Cream/tree/main/TinyCLIP),
копия MIT-лицензии — [LICENSE.child_vision.txt](models/candidates/LICENSE.child_vision.txt).
MiVOLOv2: [проект авторов](https://github.com/WildChlamydia/MiVOLO),
копия исторического Apache-2.0 LICENSE — [LICENSE.child_age.txt](models/candidates/LICENSE.child_age.txt).
Манифест [child_filter_v1.json](models/candidates/child_filter_v1.json) фиксирует
ревизии источников, SHA-256 скачанных файлов и источник каждой лицензии.
Age-only ONNX получен из отдельно указанной сторонней конвертации; официальный
исторический LICENSE не является заявлением об условиях всех нынешних версий.
Эти экспериментальные веса не включены в компактную серверную поставку.
