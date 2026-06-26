# CodeFarming

My technical blog — *"Software, ever growing."*

Published at <https://lprekon.github.io/>.

## Build

- **[Hugo](https://gohugo.io/)** static site generator. No external theme — the layouts and CSS live in this repo (`layouts/` + `assets/css/main.css`), so plain `hugo` works (the extended build / Dart Sass is no longer required).
- Config lives in `hugo.toml`. Posts live under `content/` (one directory per post, each with an `index.md`).
- Some post graphics are **generated from a Jupyter notebook** (e.g. `content/simd_benchmarking/simd_blogpost_graphics.ipynb`). CI executes the notebooks at build time, so the generated images aren't checked in.

## Running locally

The theme is in-repo, so no submodules are needed for the site to build. (One submodule remains: the SIMD post's benchmark source under `content/simd_benchmarking/`. Initialize it if you want that code present.)

```bash
git submodule update --init --recursive   # optional — only for the benchmark source
```

Start the dev server:

```bash
hugo server -D
```

Open <http://localhost:1313/>. Hugo live-reloads on save.

> **Note:** locally you'll see whatever generated images are present on disk. If a post's images come from a notebook (see above), regenerate them by running the notebook before they'll appear:
>
> ```bash
> jupyter nbconvert --to notebook --execute --inplace path/to/notebook.ipynb
> ```

To produce the final static build (output in `public/`):

```bash
hugo --gc --minify
```

## Deploying

All pushes to main trigger automatic deployment by running the github action in `.github/workflows/hugo.yaml`, which:

1. Checks out the repo with submodules,
2. Installs LaTeX + Python and executes every `.ipynb` to regenerate post graphics,
3. Builds the site with Hugo, and
4. Publishes the site to GitHub Pages

## Layout

- `layouts/_default/baseof.html` — page skeleton (head, header, footer).
- `layouts/partials/` — `head.html`, `header.html`, `footer.html`.
- `layouts/index.html` — home (sprout image + post list); `single.html` — a post; `list.html` — section pages.
- `assets/css/main.css` — the CSS. The top nav is driven by `[[menu.main]]` in `hugo.toml`.

