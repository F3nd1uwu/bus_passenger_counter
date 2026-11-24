"""One archive attempt. Invoked only by the local scheduler, never from HTTP."""
import argparse
from dataclasses import asdict, replace
import hashlib
import logging
import os
from pathlib import Path
import signal
from threading import Event, Thread
import time

from tools import archive_protocol as protocol
from tools.archive_resources import ProcessLock


class ReviewRequired(Exception):
    pass


def freeze_recipe(cfg, path):
    values = {k: v for k, v in asdict(cfg).items() if not k.startswith('DB_')
              and k not in ('CPU_THREADS', 'DETECTOR_CPU_THREADS', 'STATE_PATH', 'SYNC_INTERVAL_SEC', 'CHECKPOINT_INTERVAL_SEC')}
    assets = ['MODEL_PATH', 'REID_MODEL_PATH', 'TRACKER_PATH', 'CLOCK_MODEL_PATH']
    if cfg.STAFF_MODE == 'model':
        assets.append('VEST_MODEL_PATH')
    if cfg.CHILD_MODE == 'model':
        assets.append('CHILD_MODEL_PATH')
    values['asset_sha256'] = {k: hashlib.sha256(cfg.path(getattr(cfg, k)).read_bytes()).hexdigest() for k in assets}
    if path.exists():
        if protocol.read(path) != values:
            raise ReviewRequired('Настройки или модели изменились. Продолжение требует проверки администратора.')
    else:
        protocol.write(path, values)


def prepare_clock(cfg, output, library):
    from core.camera_clock import ClockMap, sha256
    from tools.build_clock_map import build
    def build_current():
        try:
            build(cfg, output, allow_media_fallback=True)
        except (ValueError, RuntimeError) as error:
            logging.exception('Camera clock preparation rejected: %s', error)
            raise ReviewRequired('Не удалось подтвердить начальное время записи. Нужна проверка часов камеры.') from error

    if not output.exists():
        digest = sha256(cfg.path(cfg.VIDEO_SOURCE))
        candidates = list(Path(library).glob('*.clock.json')) if library else []
        for candidate in candidates:
            if protocol.read(candidate).get('source_sha256') != digest:
                continue
            try:
                clock = ClockMap.load(candidate, cfg)
                protocol.write(output, clock.payload)
                break
            except ValueError:
                continue
        if not output.exists():
            build_current()
    try:
        return ClockMap.load(output, cfg)
    except ValueError as error:
        # Upgrade only unstarted jobs rejected by an older strict builder.
        # Preserve the old map, OCR cache and every existing counter checkpoint.
        metadata = protocol.read(output)
        if ('unreviewed' in str(error) and not cfg.path(cfg.STATE_PATH).exists()
                and metadata.get('builder_sha256') != sha256(Path(__file__).with_name('build_clock_map.py'))):
            backup = output.with_name(f'{output.stem}.before-media-fallback-{time.time_ns()}.json')
            output.rename(backup)
            logging.info('Rebuilding unstarted job clock with media fallback; previous map saved as %s', backup.name)
            build_current()
            try:
                return ClockMap.load(output, cfg)
            except ValueError as rebuilt_error:
                logging.exception('Rebuilt camera clock map rejected: %s', rebuilt_error)
                raise ReviewRequired('Новая карта времени требует проверки.') from rebuilt_error
        logging.exception('Camera clock map rejected: %s', error)
        raise ReviewRequired('Карта экранного времени требует проверки. Подсчёт не начат.') from error


def deliver_completed(cfg):
    from database.local_store import LocalStore
    from database.db_client import DatabaseClient
    from core.service import synchronize
    store = LocalStore(cfg.path(cfg.STATE_PATH))
    db = DatabaseClient(cfg.DB_HOST, cfg.DB_PORT, cfg.DB_NAME, cfg.DB_USER, cfg.DB_PASS, cfg.DB_SSLMODE)
    try:
        checkpoint = store.load()
        if not checkpoint or not checkpoint['payload'].get('completed'):
            return False
        synchronize(store, db)  # Success means final checkpoint and events committed remotely.
        return True
    finally:
        store.close()
        db.close()


def execute(spec):
    root = Path(spec['root'])
    attempt = Path(spec['attempt_dir'])
    token = spec['token']
    stop = Event()
    lease_lost = Event()
    lock = None
    progress = {'token': token, 'phase': 'starting'}
    previous = {}

    def interrupted(*_):
        raise InterruptedError('Worker stopped')

    def publish(**values):
        if not protocol.lease_valid(root / 'lease.json', token):
            raise InterruptedError('Worker lease expired')
        progress.update(values, token=token, updated=time.time())
        protocol.write(attempt / 'progress.json', progress)

    def watchdog():
        while not stop.wait(3):
            if not protocol.lease_valid(root / 'lease.json', token):
                lease_lost.set()
                os.kill(os.getpid(), signal.SIGTERM)
                return

    outcome, error = 'retry', 'Обработчик завершился до сохранения результата.'
    try:
        lock = ProcessLock(root / 'worker.lock')
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        Thread(target=watchdog, daemon=True).start()
        publish(phase='starting')
        from config.settings import Settings
        from config.profiles import apply_profile
        cfg = replace(apply_profile(Settings.from_env(), spec['profile']),
                      VIDEO_SOURCE=spec['source'], SOURCE_KIND='file', BUS_ID=spec['bus'], CAMERA_ID=spec['camera'],
                      STATE_PATH=str(root / 'counter.sqlite3'), CLOCK_MAP_PATH=str(root / 'clock.json'),
                      TIME_SOURCE='camera', VIDEO_START='', DEVICE='cpu', CPU_THREADS=spec['threads'],
                      DETECTOR_CPU_THREADS=spec['threads'], CPU_DETECTOR_BACKEND=spec['backend'])
        cfg.validate()
        freeze_recipe(cfg, root / 'recipe.json')
        publish(phase='clock')
        clock = prepare_clock(cfg, root / 'clock.json', spec.get('clock_library'))
        if lease_lost.is_set():
            raise InterruptedError
        publish(phase='processing', total_frames=clock.end_frame,
                clock_note=('Время камеры восстанавливается по времени видео между читаемыми отметками. На склейках возможна погрешность.'
                            if clock.payload.get('clock_policy') == 'camera_with_media_fallback_v1' else ''))
        from core.service import run
        def report(values):
            if values.get('totals_complete') is False:
                raise ReviewRequired('Старое состояние не содержит полного итога записи через полночь. Нужна проверка администратора.')
            publish(phase='processing', **values)
        run(cfg, on_progress=report)
        if lease_lost.is_set():
            raise InterruptedError
        publish(phase='syncing')
        if not deliver_completed(cfg):
            raise InterruptedError
        outcome, error = 'succeeded', ''
    except ReviewRequired as exception:
        outcome, error = 'review', str(exception)
    except InterruptedError:
        outcome, error = 'interrupted', 'Обработка прервана. Продолжим с сохранённого состояния.'
    except Exception as exception:
        # Detailed diagnostics stay in the private attempt log, not in the UI.
        logging.exception('Archive attempt failed (%s)', type(exception).__name__)
        outcome, error = 'retry', f'Ошибка обработчика ({type(exception).__name__}). Состояние сохранено для повторного запуска.'
    finally:
        stop.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        protocol.write(attempt / 'result.json', dict(token=token, outcome=outcome, error=error))
        if lock:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    execute(protocol.read(args.spec))


if __name__ == '__main__':
    main()
