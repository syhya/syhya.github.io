# Yue Shui's Blog

Source for [Yue Shui's Blog](https://syhya.github.io/), a personal technical blog about large language models, agents, reinforcement learning, model training, and inference systems. Articles combine research explanations, mathematical derivations, diagrams, and code examples in English and Simplified Chinese.

[English site](https://syhya.github.io/) · [中文站点](https://syhya.github.io/zh/) · [Deployment workflow](https://github.com/syhya/syhya.github.io/actions/workflows/gh-pages.yml)

## Stack and features

- **Hugo** generates the static site from Markdown and YAML configuration.
- **PaperMod**, customized and checked into this repository, provides the theme, navigation, archives, search, and light/dark modes.
- **MathJax** renders mathematical notation; **Fancybox** provides image galleries.
- **GitHub Actions and GitHub Pages** build and publish the site whenever `main` receives a push.
- **Bilingual page bundles** keep articles and their supporting images and code together.

## Run locally

Install Git and Hugo with Extended support using the [Hugo installation guide](https://gohugo.io/installation/). Hugo **0.165.0** was used by the [successful production build on September 6, 2026](https://github.com/syhya/syhya.github.io/actions/runs/34038769917); use that version to match this known working build. The workflow currently selects `latest`, so this is a verified version, not an enforced version pin.

```sh
git clone https://github.com/syhya/syhya.github.io.git
cd syhya.github.io
hugo version
hugo server -D
```

Open [the local English site](http://localhost:1313/) or [the local Chinese site](http://localhost:1313/zh/). The `-D` flag includes drafts. Add `-F` when previewing posts whose publication date is in the future:

```sh
hugo server -D -F
```

The site build does not require npm, Python, or a separately installed Go toolchain. Some article code examples have their own dependencies; consult the relevant post before running them.

The files in `themes/PaperMod/` are tracked directly by this repository. Although `.gitmodules` still contains a theme entry, the current Git tree has no theme submodule. A normal clone includes the theme; no submodule initialization is needed.

## Repository layout

```text
.
├── .github/workflows/gh-pages.yml   # Build and deployment workflow
├── archetypes/default.md           # Minimal template for new drafts
├── assets/                         # Hugo asset files
├── config.yaml                     # Site, language, navigation, and markup settings
├── content/
│   ├── en/                         # English articles, search, and archive pages
│   └── zh/                         # Chinese articles, search, and archive pages
├── layouts/                        # Site overrides, render hooks, and shortcodes
├── static/                         # Files copied to the published site
├── themes/PaperMod/                 # Vendored theme, including local customizations
├── TAXONOMY.md                      # Canonical category and tag vocabulary
└── public/                         # Generated output; ignored by Git
```

Make site configuration changes in `config.yaml`. Prefer root-level `layouts/` overrides for template changes, and preserve existing customizations when updating the vendored theme. Historical generated files also exist at the repository root; the deployment workflow publishes only `public/`.

## Build and deploy

Run the same build command used by CI:

```sh
hugo --minify
```

Hugo writes the generated site to `public/`. Commit source files and assets; keep `public/` out of source commits.

The current [.github/workflows/gh-pages.yml](.github/workflows/gh-pages.yml) runs this sequence:

```text
Push to main → Install Hugo → Build public/ → Push output to gh-pages → GitHub Pages
```

It uses `peaceiris/actions-hugo` with `hugo-version: latest` and `extended: true`, then `peaceiris/actions-gh-pages` with `GITHUB_TOKEN`. GitHub Pages is configured to publish from the root of `gh-pages`. When reproducing this setup in a fork, configure **Settings → Pages → Deploy from a branch → gh-pages → /(root)** and allow the deployment job to write repository contents. See [GitHub's publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

For a fork or a different domain, also update `baseURL`, identity, social links, and analytics settings in `config.yaml`.

## Contributing

Corrections to explanations, translations, equations, links, and code examples are welcome through [issues](https://github.com/syhya/syhya.github.io/issues) or pull requests. Reference the affected post and include a supporting source when correcting a technical claim. Keep changes focused and explain how you checked them.

## License

This repository includes an [MIT License](LICENSE). PaperMod includes its own [MIT license notice](themes/PaperMod/LICENSE). Referenced papers, quoted material, and third-party illustrations remain subject to their respective licenses and attribution requirements.
