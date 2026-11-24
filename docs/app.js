'use strict';

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

function showTab(selected, moveFocus = false) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
  }
  if (moveFocus) selected.focus();
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => showTab(tab));
  tab.addEventListener('keydown', (event) => {
    const choices = {
      ArrowRight: (index + 1) % tabs.length,
      ArrowLeft: (index + tabs.length - 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    };
    if (!Object.hasOwn(choices, event.key)) return;
    event.preventDefault();
    showTab(tabs[choices[event.key]], true);
  });
}

if (tabs.length) showTab(tabs[0]);
