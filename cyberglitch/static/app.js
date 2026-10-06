// Delegated handlers, so they keep working after htmx swaps the settings in
const current = (el) => el.type === "checkbox" ? String(el.checked) : el.value;
const same = (el) => el.type === "number"
  ? Number(el.value) === Number(el.dataset.default)
  : current(el).toLowerCase() === el.dataset.default.toLowerCase();
const ctl = document.getElementById("ctl");
ctl.addEventListener("input", (e) => {
  let el = e.target;
  const sl = el.closest(".sl");
  if (sl) {
    // Slider and number box mirror each other; the number box is the submitted value
    const [range, num] = sl.querySelectorAll("input");
    if (el === range) num.value = range.value;
    else {
      if (+num.value > +range.max) range.max = num.value;
      if (+num.value < +range.min) range.min = num.value;
      range.value = num.value;
    }
    el = num;
  }
  if (el.dataset.default !== undefined) el.closest("label").classList.toggle("changed", !same(el));
});
ctl.addEventListener("click", (e) => {
  if (!e.target.classList.contains("rst")) return;
  const el = e.target.parentElement.querySelector("[data-default]");
  if (el.type === "checkbox") el.checked = el.dataset.default === "true";
  else el.value = el.dataset.default;
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
document.getElementById("reset-all").addEventListener("click", (e) => {
  // Back to the starting look: re-selecting it fires the preset's own /fields request
  document.querySelectorAll("#presets input").forEach((c) => (c.checked = false));
  const look = document.querySelector(`#presets input[type=radio][value="${e.target.dataset.look}"]`);
  look.checked = true;
  look.dispatchEvent(new Event("change", { bubbles: true }));
});
