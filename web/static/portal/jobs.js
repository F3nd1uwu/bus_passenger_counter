(() => {
  const list = document.querySelector('[data-jobs-url]');
  if (!list) return;
  const rows = Array.from(list.querySelectorAll('[data-upload-id]'));
  const service = document.querySelector('[data-dispatcher-state]');
  const poll = async () => {
    if (document.hidden) return;
    try {
      const response = await fetch(`${list.dataset.jobsUrl}?ids=${rows.map(r => r.dataset.uploadId).join(',')}`, {credentials: 'same-origin'});
      if (!response.ok || response.redirected) return;
      const data = await response.json();
      service.textContent = data.dispatcher.online ? 'Служба обработки работает.' :
        'Служба обработки пока недоступна. Записи сохраняются в очереди.';
      if (data.dispatcher.online && data.dispatcher.active !== undefined) {
        service.textContent += ` Обработчиков занято: ${data.dispatcher.active} из ${data.dispatcher.capacity}.`;
      }
      for (const job of data.jobs) {
        const row = rows.find(r => r.dataset.uploadId === job.id);
        const label = row?.querySelector('[data-job-label]');
        if (!label) continue;
        label.textContent = job.label;
        row.querySelector('[data-job-phase]').textContent = job.phase +
          (job.phase_code === 'processing' ? ` · ${job.percent}% · ${job.fps} кадр/с` : '');
        row.querySelector('[data-job-error]').textContent = job.error;
        const clockNote = row.querySelector('[data-job-clock-note]');
        if (clockNote) clockNote.textContent = job.clock_note || '';
        row.querySelector('[data-job-counts]').textContent = `Входы: ${job.counted_in} · Выходы: ${job.counted_out}`;
        const progress = row.querySelector('[data-job-progress]');
        progress.value = job.percent;
        progress.hidden = !job.percent && job.status !== 'running';
        const retry = row.querySelector('[data-job-retry]');
        if (retry) retry.hidden = !['review', 'failed'].includes(job.status);
        const download = row.querySelector('[data-job-download]');
        if (download) download.hidden = job.status !== 'succeeded';
      }
    } catch (_) {
      service.textContent = 'Не удалось обновить состояние. Повторим при восстановлении соединения.';
    }
  };
  setInterval(poll, 10000);
})();
