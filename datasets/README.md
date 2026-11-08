# Локальные данные обучения

Изображения пассажиров, исходное видео, разметка и архивы датасета не публикуются
в Git. При наличии локального датасета используйте структуру:

```text
datasets/bus_people/
  annotations.json
  manifest.json
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

Формат YOLO, проверка разметки и обучение описаны в
[руководстве разработчика](../docs/DEVELOPMENT.md).
