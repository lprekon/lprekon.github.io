// Three-way color theme control: System -> Light -> Dark -> System.
// The chosen preference is stored in localStorage and applied via the
// data-theme attribute on <html>. When set to "system" the attribute is
// removed so the CSS falls back to prefers-color-scheme.
(function () {
  var KEY = "theme";
  var ORDER = ["system", "light", "dark"];
  var LABEL = { system: "System", light: "Light", dark: "Dark" };
  var ICON = { system: "◐", light: "☀", dark: "☽" }; // ◐ ☀ ☾

  function current() {
    var v = localStorage.getItem(KEY);
    return ORDER.indexOf(v) === -1 ? "system" : v;
  }

  function apply(pref) {
    if (pref === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", pref);
    }
  }

  function render(btn, pref) {
    btn.textContent = ICON[pref];
    btn.title = "Theme: " + LABEL[pref] + " (click to change)";
    btn.setAttribute("aria-label", "Color theme: " + LABEL[pref]);
  }

  var btn = document.querySelector(".theme-toggle");
  if (!btn) return;

  render(btn, current());

  btn.addEventListener("click", function () {
    var next = ORDER[(ORDER.indexOf(current()) + 1) % ORDER.length];
    localStorage.setItem(KEY, next);
    apply(next);
    render(btn, next);
  });
})();
