(() => {
  const form = document.querySelector('#video-upload-form');
  if (!form) return;
  const token = form.querySelector('[name=csrfmiddlewaretoken]').value;
  const output = document.querySelector('#upload-progress');
  const pause = document.querySelector('#upload-pause');
  const submit = form.querySelector('[type=submit]');
  let stopped = false;
  let active = false;
  let controller;
  const show = (message) => { output.textContent = message; };
  const url = (id, action) => `${form.dataset.createUrl.replace(/create\/$/, '')}${id}/${action}/`;
  async function api(path, body, extra = {}) {
    controller = new AbortController();
    const response = await fetch(path, {method: 'POST', credentials: 'same-origin', signal: controller.signal,
      headers: {'X-CSRFToken': token, ...extra}, body});
    if (response.redirected) throw new Error('Войдите в кабинет заново, затем выберите исходные файлы для продолжения.');
    if (!response.headers.get('Content-Type')?.includes('application/json')) {
      throw new Error('Сервер не принял запрос. Обновите страницу и продолжите загрузку.');
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось загрузить запись.');
    return data;
  }
  async function fingerprint(file) {
    // WebCrypto needs HTTPS or localhost. Reading two slices bounds browser memory.
    if (!window.crypto?.subtle) throw new Error('Для загрузки откройте кабинет по HTTPS или через localhost.');
    const bytes = new Uint8Array(await new Blob([file.slice(0, 1048576), file.slice(Math.max(0, file.size - 1048576))]).arrayBuffer());
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2, '0')).join('');
  }
  pause.addEventListener('click', () => { stopped = true; controller?.abort(); });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (active) return;
    active = true; stopped = false;
    const files = Array.from(form.elements.videos.files);
    const bus = form.elements.bus.value;
    const camera = form.elements.camera_id.value;
    submit.disabled = true; pause.hidden = false;
    form.elements.bus.disabled = form.elements.camera_id.disabled = form.elements.videos.disabled = true;
    try {
      let completed = 0;
      for (const file of files) {
        if (stopped) break;
        show(`Проверяем файл: ${file.name}`);
        const signature = await fingerprint(file);
        if (stopped) break;
        let upload = await api(form.dataset.createUrl, JSON.stringify({name: file.name, size: file.size,
          bus, camera_id: camera, fingerprint: signature}), {'Content-Type': 'application/json'});
        while (upload.received < file.size) {
          if (stopped) break;
          show(`${file.name}: ${Math.floor(upload.received / file.size * 100)}% · ${completed + 1} из ${files.length}`);
          const offset = upload.received;
          const chunk = file.slice(offset, Math.min(offset + upload.chunk_size, file.size));
          upload = await api(url(upload.id, 'chunk'), chunk,
            {'Content-Type': 'application/octet-stream', 'Upload-Offset': String(offset)});
        }
        if (stopped) break;
        show(`${file.name}: проверяем загруженное видео…`);
        await api(url(upload.id, 'finish'), '');
        completed += 1;
      }
      if (stopped) show('Загрузка приостановлена. Выберите те же файлы и нажмите «Загрузить видео», чтобы продолжить.');
      else { active = false; show(`Загружено записей: ${completed}. Обновляем список…`); window.location.reload(); }
    } catch (error) {
      show(error.name === 'AbortError' ? 'Загрузка приостановлена. Уже принятые части сохранены.' :
        `${error.message} Принятые части сохранены; для продолжения выберите те же файлы.`);
    } finally {
      active = false; pause.hidden = true; submit.disabled = false; submit.removeAttribute('aria-busy');
      form.elements.bus.disabled = form.elements.camera_id.disabled = form.elements.videos.disabled = false;
    }
  });
  window.addEventListener('beforeunload', (event) => { if (active) { event.preventDefault(); event.returnValue = ''; } });
  document.querySelectorAll('[data-upload-cancel]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (active) { show('Сначала приостановите текущую загрузку.'); return; }
      button.disabled = true;
      try { await api(button.dataset.uploadCancel, ''); window.location.reload(); }
      catch (error) { show(error.message); button.disabled = false; }
    });
  });
})();
