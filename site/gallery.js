"use strict";

const search = document.querySelector("#search");
const framework = document.querySelector("#framework");
const severity = document.querySelector("#severity");
const count = document.querySelector("#count");
const cards = Array.from(document.querySelectorAll(".example"));
const empty = document.querySelector("#empty");
const copyStatus = document.querySelector("#copy-status");

function filterExamples() {
  const query = search.value.trim().toLowerCase();
  let visible = 0;
  for (const card of cards) {
    const matches = (!framework.value || card.dataset.framework === framework.value)
      && (!severity.value || card.dataset.severity === severity.value)
      && card.textContent.toLowerCase().includes(query);
    card.hidden = !matches;
    if (matches) visible += 1;
  }
  count.textContent = `${visible} of ${cards.length} examples`;
  empty.hidden = visible !== 0;
}

function resetFilters() {
  search.value = "";
  framework.value = "";
  severity.value = "";
  filterExamples();
}

function revealLinkedExample() {
  const card = cards.find(item => `#${item.id}` === window.location.hash);
  if (card) {
    resetFilters();
    card.scrollIntoView({ block: "start" });
  }
}

document.querySelector(".filters").hidden = false;
search.addEventListener("input", filterExamples);
framework.addEventListener("change", filterExamples);
severity.addEventListener("change", filterExamples);
document.querySelector("#reset").addEventListener("click", () => {
  resetFilters();
  search.focus();
});
window.addEventListener("hashchange", revealLinkedExample);
for (const button of document.querySelectorAll("[data-copy]")) {
  button.hidden = false;
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(code.textContent);
      copyStatus.textContent = `${button.getAttribute("aria-label")}: copied.`;
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy code"; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Code selected. Use your browser's Copy command.";
    }
  });
}
filterExamples();
revealLinkedExample();
