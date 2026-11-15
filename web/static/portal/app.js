const toggle = document.querySelector('[data-menu]');
if (toggle) {
  const closeMenu = () => {
    toggle.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('menu-open');
    toggle.focus();
  };
  document.querySelector('[data-menu-close]')?.addEventListener('click', closeMenu);
  toggle.addEventListener('click', () => {
    const opened = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(opened));
    document.body.classList.toggle('menu-open', opened);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeMenu();
    }
  });
}
document.querySelectorAll('form').forEach((form) => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (button) { button.disabled = true; button.setAttribute('aria-busy', 'true'); }
  });
});
