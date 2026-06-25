# CodeFarming

My technical blog — *"Software, ever growing."*

Published at <https://lprekon.github.io/>.

## Build

- **[Hugo](https://gohugo.io/)** static site generator (use the **extended** build — the theme needs Dart Sass).
- Theme: **[paige](https://github.com/willfaught/paige)**, vendored as a git submodule.
- Config lives in `hugo.toml`. Posts live under `content/` (one directory per post, each with an `index.md`).
- Some post graphics are **generated from a Jupyter notebook** (e.g. `content/simd_benchmarking/simd_blogpost_graphics.ipynb`). CI executes the notebooks at build time, so the generated images aren't checked in.

## Running locally

This repo uses git submodules. To initialize:

```bash
git submodule update --init --recursive
```

start the dev server:

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

