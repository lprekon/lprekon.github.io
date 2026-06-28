// Copy each footnote's text onto its reference so it can show as a hover tooltip.
document.querySelectorAll("a.footnote-ref").forEach(function (ref) {
  var note = document.getElementById(ref.getAttribute("href").slice(1)); // <li id="fn:N">
  if (!note) return;
  var clone = note.cloneNode(true);
  clone.querySelectorAll(".footnote-backref").forEach(function (b) { b.remove(); });
  ref.dataset.footnote = clone.textContent.trim();
  ref.classList.add("has-footnote-tooltip");
});
