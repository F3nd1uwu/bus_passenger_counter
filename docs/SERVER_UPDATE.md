# Обновление существующего сервера

## Обновление действующего веб-кабинета и очереди

Если видео уже загружаются через кабинет, используйте этот сценарий. Команды
ниже для прежних отдельных контейнеров `video-test-*` к такой установке не относятся.

```bash
cd /opt/bus_passenger_counter
docker compose -f docker-compose.yml -f docker-compose.web.yml -f docker-compose.archive.yml stop -t 120 archive_scheduler web
git status --short
git pull --ff-only origin server
python3 -m tools.verify_release
python3 -m tools.start_portal
```

Для уже настроенного HTTPS добавьте `--https` к последней команде. При временном
доступе через SSH-туннель этот флаг не нужен. Не добавляйте `--skip-build` при
обновлении: новые код и миграции должны попасть в образы. Установщик сохраняет
существующие настройки и аккаунты и применяет миграции перед запуском очереди.
Загруженные видео и постоянные тома остаются на месте. Перед обновлением сохраняйте
резервные копии PostgreSQL и `web_state` по инструкции кабинета.

После обновления режима часов нажмите «Повторить обработку» у записей со статусом
«Требуется проверка». Непроверенная старая карта без начатого состояния подсчёта
будет сохранена отдельно и перестроена по кешу OCR. Уже начатые состояния и
пригодные карты автоматически не заменяются. Восстановление времени по PTS
описано в [ARCHIVE_QUEUE.md](ARCHIVE_QUEUE.md).

## Прежний запуск отдельных обработчиков

Инструкция для каталога `/opt/bus_passenger_counter` и прежних контейнеров
`video-test-1`, `video-test-2`. Команды выполняются по SSH на сервере, кроме
отдельно отмеченной команды туннеля на ноутбуке. Сохраняйте каталог и имя
Compose-проекта: от него зависят имена сети и постоянных томов.

Для обновления приложения перезагружать VPS не требуется. Не используйте
`docker compose down -v`: эта команда удаляет постоянные данные.

## Сохранить прежний прогон

Остановите обработчики штатно. PostgreSQL остаётся работать:

```bash
cd /opt/bus_passenger_counter
backup=".local/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup"

docker stop -t 120 video-test-1 video-test-2
docker logs video-test-1 > "$backup/video-test-1.log" 2>&1
docker logs video-test-2 > "$backup/video-test-2.log" 2>&1
docker cp video-test-1:/app/state "$backup/state-video-1"
docker cp video-test-2:/app/state "$backup/state-video-2"
cp .env "$backup/.env"
cp -a models "$backup/models"
git rev-parse HEAD > "$backup/commit.txt"
docker compose exec -T postgres sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$backup/postgres.dump"
```

Если есть `.env.web`, сохраните и его. Если кабинет уже используется, сохраните
том `web_state` после штатной остановки web, по инструкции [WEB.md](WEB.md).
Не удаляйте старые контейнеры и тома. Копии содержат приватные данные и не
предназначены для Git или публичной пересылки.

## Получить обновление

```bash
git status --short
git fetch origin
git switch server
git pull --ff-only origin server
python3 -m tools.verify_release
```

Если Git сообщает о локальных изменениях или расходящейся истории, остановите
обновление и разберите их. Принудительный сброс здесь не нужен. Реальные `.env`,
видео, карты времени и базы не входят в Git. Проверка манифеста выполняется
до локальной адаптации исходников и профиля.

## Видео и часы камеры

В `data` должны находиться соответствующие друг другу файлы:

```text
data/test.mp4
data/test.clock.json
data/test2.mp4
data/test2.clock.json
```

Подготовленные приватные карты переносятся отдельно с ноутбука. Повторная
загрузка неизменённых видео не требуется. Карта проверяется по содержимому
видеофайла и модели OCR; карта другого файла отклоняется.

Передача без командной строки: SFTP-клиент WinSCP, адрес VPS, порт 22, тот же
пользователь и способ авторизации, что для SSH. Каталог справа —
`/opt/bus_passenger_counter/data`. Не заменяйте видео во время обработки.

В `.env` сохраните прежние реквизиты PostgreSQL и задайте:

```dotenv
SOURCE_KIND=file
TIME_SOURCE=camera
TIMEZONE=Europe/Saratov
CHECKPOINT_INTERVAL_SEC=30
FRAME_STRIDE=1
CPU_DETECTOR_BACKEND=onnx
DETECTOR_CPU_THREADS=0
```

Значение часового пояса относится к текущим подготовленным картам. Для другой
камеры/пояса карту готовят с соответствующими параметрами по README.

## Запустить веб-кабинет

```bash
python3 -m tools.setup_web_env
docker compose -f docker-compose.yml -f docker-compose.web.yml build counter_service web
docker compose up -d postgres
docker compose -f docker-compose.yml -f docker-compose.web.yml run --rm --no-deps web_setup
docker compose -f docker-compose.yml -f docker-compose.web.yml up -d --no-deps web
```

`setup_web_env` сохраняет существующую `.env.web`. Для нестандартных параметров
БД проверьте `WEB_ANALYTICS_DB_NAME` и прочие `WEB_*`; `WEB_TIMEZONE` должен
совпадать с бизнес-поясом счётчика. Секреты через чат или Git не передаются.

Если администратора ещё нет:

```bash
docker compose -f docker-compose.yml -f docker-compose.web.yml exec web python manage_web.py createsuperuser
```

Интерактивно задайте логин и пароль. Не запускайте просто `docker compose up -d`
при двух отдельных видео: это дополнительно запустит стандартный `counter_service`.
Команды выше запускают только PostgreSQL и web.

## Два независимых видео

Первый новый серверный тест использует ONNX, по два потока на обработчик.
Новые тестовые BUS_ID и файлы состояния отделяют результаты от старого прогона.
Старую память нельзя переносить на новые часы камеры/модели. Параметр STATE_PATH
в `docker compose run -e` переопределяет путь из Compose, который иначе имеет
приоритет над значением в `.env`.

```bash
docker compose run -d --no-deps --name archive-v2-1 \
  -e BUS_ID=TEST_V2_VIDEO_1 \
  -e STATE_PATH=/app/state/archive-v2-video-1.sqlite3 \
  counter_service python main.py --profile test2 \
  --source /app/data/test.mp4 --time-source camera \
  --clock-map /app/data/test.clock.json --cpu-threads 2 \
  --cpu-detector-backend onnx

docker compose run -d --no-deps --name archive-v2-2 \
  -e BUS_ID=TEST_V2_VIDEO_2 \
  -e STATE_PATH=/app/state/archive-v2-video-2.sqlite3 \
  counter_service python main.py --profile test2 \
  --source /app/data/test2.mp4 --time-source camera \
  --clock-map /app/data/test2.clock.json --cpu-threads 2 \
  --cpu-detector-backend onnx
```

Команды создания контейнеров выполняются один раз. При остановке без изменения
конфигурации используйте `docker start archive-v2-1 archive-v2-2`: обработчики
продолжат с checkpoint. Завершённый файл с прежним состоянием повторно не считается.
Для другого файла нужен отдельный STATE_PATH; ID между файлами не переносятся.

```bash
docker ps -a
docker logs --tail 30 archive-v2-1
docker logs --tail 30 archive-v2-2
docker stats --no-stream
```

Новые логи показывают FPS, позицию `source_frame` и камерное время. После
перезапуска счётчик `Processed` начинается заново; общий прогресс определяется
по `source_frame`. `docker logs -f archive-v2-1` включает наблюдение;
`Ctrl+C` закрывает просмотр логов, контейнер продолжает работать.

OpenVINO остаётся опциональным экспериментом и требует образа с установленным
пакетом. Его локальное ускорение не является замером Intel VPS. Отдельные
ошибки подсчёта сохраняются у обоих движков. Полные протоколы и команды сравнения:
[CPU-проверки полной версии](https://github.com/F3nd1uwu/bus_passenger_counter/blob/diploma/docs/CPU_OPTIMIZATION.md).

## Открыть сайт с ноутбука

На ноутбуке замените `IP_СЕРВЕРА` адресом VPS и выполните:

```powershell
ssh -N -L 8000:127.0.0.1:8000 root@IP_СЕРВЕРА
```

Пока туннель открыт, сайт доступен по http://127.0.0.1:8000/.
Сервер обрабатывает видео независимо от ноутбука. При следующем просмотре
создайте туннель заново. В кабинете выбирайте даты записи — 1 и 9 ноября 2026,
а не дату запуска сервера. Зарегистрируйте новые BUS_ID и назначьте сотрудникам.

Кабинет пока не принимает видео и не управляет работниками. Доступны просмотр
статистики, роли и проверочные CSV; загрузка файлов выполняется через SFTP.
Обмен с 1С не настроен. Подробности: [WEB.md](WEB.md).

## Перезагрузка VPS при необходимости

```bash
docker stop -t 120 archive-v2-1 archive-v2-2
reboot
```

После повторного SSH-подключения:

```bash
cd /opt/bus_passenger_counter
docker compose -f docker-compose.yml -f docker-compose.web.yml up -d postgres web
docker start archive-v2-1 archive-v2-2
```

Продолжение требует тех же видео, моделей, настроек и постоянных томов.
