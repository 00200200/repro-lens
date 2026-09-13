"use strict";

const search = document.querySelector("#search");
const framework = document.querySelector("#framework");
const severity = document.querySelector("#severity");
const count = document.querySelector("#count");
const cards = Array.from(document.querySelectorAll(".example"));
const links = Array.from(document.querySelectorAll(".case-nav a"));
const empty = document.querySelector("#empty");
const copyStatus = document.querySelector("#copy-status");
let selected = cards.find(card => `#${card.id}` === location.hash)?.id
  || cards.find(card => card.id === "xgboost-linear-updater")?.id || cards[0].id;
let toastTimer;

function filterExamples() {
  const query = search.value.trim().toLowerCase();
  const matches = cards.filter(card => (!framework.value || card.dataset.framework === framework.value)
    && (!severity.value || card.dataset.severity === severity.value)
    && card.textContent.toLowerCase().includes(query));
  if (!matches.some(card => card.id === selected)) selected = matches[0]?.id;
  // A URL naming a case must follow the visible case when filters change it.
  if (cards.some(card => `#${card.id}` === location.hash)) {
    const hash = selected ? `#${selected}` : "#examples";
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }
  for (const card of cards) card.hidden = card.id !== selected;
  for (const link of links) {
    link.hidden = !matches.some(card => card.id === link.dataset.case);
    if (link.dataset.case === selected) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  }
  count.textContent = `${matches.length} of ${cards.length} examples`;
  empty.hidden = matches.length !== 0;
  document.querySelector("#selection-status").textContent = selected
    ? `Viewing ${cards.find(card => card.id === selected).querySelector("h2").textContent}.`
    : "No example selected.";
  const navigation = document.querySelector(".case-nav");
  const activeLink = links.find(link => link.dataset.case === selected);
  if (activeLink && navigation.scrollWidth > navigation.clientWidth) {
    // Keep the selected mobile card visible without scrolling the whole page.
    navigation.scrollLeft = activeLink.offsetLeft - (navigation.clientWidth - activeLink.offsetWidth) / 2;
  }
}

function resetFilters() {
  search.value = "";
  framework.value = "";
  severity.value = "";
  filterExamples();
}

function revealLinkedExample() {
  const card = cards.find(item => `#${item.id}` === location.hash);
  if (card) {
    selected = card.id;
    resetFilters();
    card.scrollIntoView({ block: "start" });
  }
}

for (const link of links) {
  link.addEventListener("click", event => {
    // Preserve native open-in-new-tab and modified-click behavior.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    selected = link.dataset.case;
    if (location.hash !== link.hash) history.pushState(null, "", link.hash);
    filterExamples();
  });
}

document.querySelector(".filters").hidden = false;
document.querySelector(".case-sidebar").hidden = false;
document.documentElement.classList.add("enhanced");
search.addEventListener("input", filterExamples);
framework.addEventListener("change", filterExamples);
severity.addEventListener("change", filterExamples);
document.querySelector("#reset").addEventListener("click", () => {
  resetFilters();
  search.focus();
});
window.addEventListener("hashchange", revealLinkedExample);
window.addEventListener("popstate", revealLinkedExample);

for (const button of document.querySelectorAll("[data-copy]")) {
  const label = button.textContent;
  let labelTimer;
  button.hidden = false;
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    clearTimeout(toastTimer);
    try {
      await navigator.clipboard.writeText(code.textContent);
      copyStatus.textContent = `${button.getAttribute("aria-label")}: copied.`;
      button.textContent = "Copied ✓";
      clearTimeout(labelTimer);
      labelTimer = setTimeout(() => { button.textContent = label; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Code selected. Use your browser's Copy command.";
    }
    toastTimer = setTimeout(() => { copyStatus.textContent = ""; }, 4000);
  });
}

const scenarioButtons = Array.from(document.querySelectorAll("[data-scenario]"));
function selectScenario(name) {
  for (const button of scenarioButtons) {
    const active = button.dataset.scenario === name;
    button.setAttribute("aria-pressed", String(active));
    document.getElementById(`demo-${button.dataset.scenario}`).hidden = !active;
  }
}
for (const button of scenarioButtons) {
  button.setAttribute("aria-controls", `demo-${button.dataset.scenario}`);
  button.addEventListener("click", () => selectScenario(button.dataset.scenario));
}
document.querySelector(".scenario-buttons").hidden = false;
selectScenario("threshold");
filterExamples();
revealLinkedExample();
