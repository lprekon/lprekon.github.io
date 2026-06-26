(function () {
  function open(src, alt) {
    var box = document.createElement("div");
    box.className = "lightbox";
    var img = document.createElement("img");
    img.src = src;
    img.alt = alt || "";
    box.appendChild(img);

    function close() {
      box.classList.remove("is-open");
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      setTimeout(function () { box.remove(); }, 150);
    }
    function onKey(e) { if (e.key === "Escape") close(); }

    box.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(box);
    document.body.style.overflow = "hidden";
    requestAnimationFrame(function () { box.classList.add("is-open"); });
  }

  document.addEventListener("click", function (e) {
    var t = e.target;
    if (t.tagName === "IMG" && t.classList.contains("zoomable")) {
      open(t.currentSrc || t.src, t.alt);
    }
  });
})();
