// Clone each footnote's markup into a tooltip on its reference, so hovering a
// footnote number shows the note itself. This has to be real DOM rather than a
// CSS `content: attr(...)` string: the notes contain KaTeX (and links), which
// only render as markup.
document.querySelectorAll("a.footnote-ref").forEach(function (ref) {
  var note = document.getElementById(ref.getAttribute("href").slice(1)); // <li id="fn:N">
  if (!note) return;

  var clone = note.cloneNode(true);
  clone.querySelectorAll(".footnote-backref").forEach(function (b) { b.remove(); });

  var tip = document.createElement("span");
  tip.className = "footnote-tooltip";
  tip.setAttribute("role", "tooltip");
  // Unwrap the note's <p>s: the tooltip lives inside a <sup>, where block
  // elements aren't valid. Each paragraph becomes a block-styled span.
  Array.prototype.forEach.call(clone.childNodes, function (node) {
    if (node.nodeName === "P") {
      var para = document.createElement("span");
      para.className = "footnote-tooltip-p";
      while (node.firstChild) para.appendChild(node.firstChild);
      tip.appendChild(para);
    } else {
      tip.appendChild(node.cloneNode(true));
    }
  });

  // Hang the tooltip off the wrapping <sup>, not the <a> — an <a> may not
  // contain another <a>, and footnotes can hold links.
  var host = ref.parentElement && ref.parentElement.tagName === "SUP" ? ref.parentElement : ref;
  host.classList.add("has-footnote-tooltip");
  host.appendChild(tip);
});
