# CLAUDE.md - AI Assistant Guide for syhya.github.io

## Project Overview

This is a multilingual technical blog built with Hugo static site generator, focusing on Large Language Models (LLM), AI, and Deep Learning topics. The blog is authored by Yue Shui and deployed to GitHub Pages at https://syhya.github.io/.

**Key Focus Areas:**
- LLM algorithms and training
- Agentic RL (Reinforcement Learning)
- Model training, RAG, and LLM Agents
- Deep Learning applications in finance, audit, and code generation

## Repository Structure

```
syhya.github.io/
├── .github/
│   └── workflows/
│       └── gh-pages.yml          # GitHub Actions deployment workflow
├── archetypes/
│   └── default.md                # Template for new posts
├── assets/
│   └── css/                      # Custom CSS files
├── content/
│   ├── en/                       # English content
│   │   ├── posts/               # Blog posts in English
│   │   ├── archives.md          # Archives page
│   │   └── search.md            # Search page
│   └── zh/                       # Chinese (简体中文) content
│       ├── posts/               # Blog posts in Chinese
│       ├── archives.md
│       └── search.md
├── layouts/
│   ├── _default/
│   │   └── _markup/
│   │       └── render-image.html # Custom image rendering
│   ├── partials/
│   │   └── extend_head.html      # Custom head content (MathJax, Fancybox)
│   └── shortcodes/
│       └── figure.html           # Custom figure shortcode with Fancybox
├── static/
│   └── css/                      # Static CSS assets
├── themes/
│   └── PaperMod/                 # Hugo PaperMod theme (git submodule)
├── config.yaml                   # Main Hugo configuration
├── .gitignore                    # Git ignore rules
├── .gitmodules                   # Git submodule configuration
└── LICENSE                       # Repository license
```

## Content Organization

### Blog Post Structure

Each blog post follows this directory structure:

```
content/{language}/posts/{date}-{topic}/
├── index.md                      # Main content file
├── image1.png                    # Post images
├── image2.png
└── ...
```

**Example:**
- `content/en/posts/2025-11-19-scaling-law/index.md`
- `content/zh/posts/2025-11-19-scaling-law/index.md`

### Post Front Matter Template

All posts use YAML front matter with the following structure:

```yaml
---
title: "Post Title"
date: 2025-11-19T12:00:00+08:00
author: "Yue Shui"
tags: ["Tag1", "Tag2", "Tag3"]
categories: ["Technical Blog"]  # or ["技术博客"] for Chinese
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true                        # Enable for posts with LaTeX math
---
```

## Key Conventions and Guidelines

### 1. Multilingual Content

- **Always create parallel content** in both English (`content/en/`) and Chinese (`content/zh/`)
- Use identical file structure and naming for both languages
- Post dates should match across languages
- Use appropriate category names: "Technical Blog" (EN) / "技术博客" (ZH)

### 2. Post Naming Convention

Format: `{YYYY}-{MM}-{DD}-{topic-slug}/index.md`

Examples:
- `2025-11-19-scaling-law/`
- `2025-08-24-gpt5/`
- `2025-03-27-llm-agent/`

**Rules:**
- Date format: YYYY-MM-DD
- Use lowercase for directory names
- Use hyphens (-) to separate words
- Keep topic names concise and descriptive

### 3. Image Handling

**Location:** Place all images in the same directory as `index.md`

**Usage in Markdown:**
```markdown
{{< figure
    src="image_name.png"
    caption="Fig. 1. Description here. (Image source: [Author](URL))"
    align="center"
    width="100%"
>}}
```

**Features:**
- Custom `figure` shortcode with Fancybox integration for image zoom
- Always include descriptive captions with figure numbers
- Include image sources when applicable
- Use relative paths (just filename)

### 4. Mathematical Content

**Enable Math Support:**
Set `math: true` in front matter for posts with LaTeX equations

**Inline Math:** Use `\( ... \)` delimiters
```markdown
This is inline math: \( E = mc^2 \)
```

**Block Math:** Use `$$` delimiters
```markdown
$$
L(N) = \left(N_{\mathrm{c}} / N\right)^{\alpha_N}
$$
```

### 5. Citations and References

**Format:**
```markdown
[Author et al., Year](https://arxiv.org/abs/xxxx.xxxxx)
```

**Examples:**
- `[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)`
- `[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)`

### 6. Draft Posts

Posts in `.gitignore`:
- Draft posts are excluded from build
- Located in paths like `content/zh/posts/2025-XX-XX-topic/`
- Keep backup versions with `-backup` suffix

**To publish a draft:**
1. Remove from `.gitignore`
2. Ensure `draft: false` in front matter
3. Verify both language versions are complete

## Development Workflow

### Creating a New Blog Post

1. **Create directory structure:**
   ```bash
   mkdir -p content/en/posts/2025-MM-DD-topic-name
   mkdir -p content/zh/posts/2025-MM-DD-topic-name
   ```

2. **Create index.md files with proper front matter**

3. **Add images to post directories**

4. **Test locally:**
   ```bash
   hugo server -D  # Include drafts
   hugo server     # Production build
   ```

5. **Commit changes:**
   ```bash
   git add content/
   git commit -m "Add new post: topic name"
   git push origin main
   ```

### Editing Existing Posts

Common edit patterns from git history:
- "Fix format issue" - Typography, spacing, or layout corrections
- "Update [topic] blog" - Content updates or additions
- "Enhance [topic] blog" - Quality improvements
- "Refine url" - Link corrections
- "Add citation part" - Adding references

**When editing:**
1. Read both language versions to ensure consistency
2. Preserve existing formatting patterns
3. Update both EN and ZH versions simultaneously
4. Test math rendering if equations are modified
5. Verify image paths remain correct

### Theme Customization

**PaperMod Theme (Git Submodule):**
- Location: `themes/PaperMod/`
- Repository: https://github.com/syhya/hugo-PaperMod.git
- **Do not modify theme files directly**
- Override with custom layouts in `/layouts/`

**Custom Overrides:**
- `/layouts/partials/extend_head.html` - Adds MathJax and Fancybox
- `/layouts/shortcodes/figure.html` - Custom figure with Fancybox
- `/layouts/_default/_markup/render-image.html` - Custom image rendering

### Updating the Theme

```bash
cd themes/PaperMod
git pull origin master
cd ../..
git add themes/PaperMod
git commit -m "Update PaperMod theme"
```

## Build and Deployment

### Local Development

```bash
# Start development server with drafts
hugo server -D

# Start production server
hugo server

# Build site (output to public/)
hugo --minify
```

### GitHub Actions Deployment

**Workflow:** `.github/workflows/gh-pages.yml`

**Trigger:** Push to `main` branch

**Process:**
1. Checkout repository (with submodules)
2. Setup Hugo (latest extended version)
3. Build site with `hugo --minify`
4. Deploy to `gh-pages` branch
5. GitHub Pages serves from `gh-pages` branch

**Important:**
- Never push directly to `gh-pages` branch
- All changes should go through `main` branch
- Build artifacts (`public/`) are gitignored

## Configuration (config.yaml)

### Critical Settings

- **baseURL:** `https://syhya.github.io/`
- **theme:** `["PaperMod"]`
- **publishDir:** `public`
- **languageCode:** `en-us` (default)
- **enableEmoji:** `true`
- **enableInlineShortcodes:** `true`

### Language Configuration

Two languages configured: English (en) and Chinese (zh)

**English (weight: 1):**
- languageCode: `en-us`
- contentDir: `content/en`

**Chinese (weight: 2):**
- languageCode: `zh-cn`
- contentDir: `content/zh`
- hasCJKLanguage: `true`

### Features Enabled

- **Math rendering:** MathJax with passthrough delimiters
- **Fancybox:** Image galleries and zoom
- **Search:** JSON output for Fuse.js search
- **RSS:** Enabled for home page
- **Google Analytics:** Tracking ID configured
- **Table of Contents:** Auto-generated, collapsible
- **Code copy buttons:** Enabled
- **Reading time & word count:** Displayed

## Common Tasks for AI Assistants

### Adding a New Post

```bash
# Create bilingual post structure
mkdir -p content/en/posts/2025-XX-XX-new-topic
mkdir -p content/zh/posts/2025-XX-XX-new-topic

# Create index.md files with proper front matter
# Add images to post directories
# Ensure math: true if LaTeX is needed
```

### Fixing Formatting Issues

Check for:
- Inconsistent spacing around headers
- Math delimiter issues (use `\(` `\)` for inline, `$$` for block)
- Image shortcode syntax
- Citation format consistency
- Front matter completeness

### Updating Images

1. Place images in post directory (same as index.md)
2. Use relative filenames in `figure` shortcode
3. Always include captions with figure numbers
4. Verify images exist in both language versions

### Adding Math Equations

1. Set `math: true` in front matter
2. Use LaTeX syntax with proper delimiters
3. Test rendering locally before committing
4. Ensure equations render correctly in both languages

### Syncing Bilingual Content

When updating one language:
1. Check if corresponding version exists in other language
2. Apply equivalent changes maintaining style
3. Verify dates and tags match
4. Keep technical terms consistent (or appropriately localized)

## Best Practices

### Content Quality

1. **Technical Accuracy:** Posts focus on LLM, AI, and Deep Learning - ensure technical correctness
2. **Citations:** Always cite sources for research papers and figures
3. **Consistency:** Maintain parallel structure between EN and ZH versions
4. **Images:** Include descriptive captions with figure numbers and sources

### Git Workflow

1. **Atomic Commits:** One logical change per commit
2. **Descriptive Messages:** Clear, concise commit messages
3. **Test Before Push:** Always build locally first
4. **Branch Strategy:** Use feature branches for major changes

### File Management

1. **No Binary Clutter:** Ignore macOS `.DS_Store`, logs, CSV files
2. **Organized Assets:** Keep images with their posts
3. **Draft Management:** Use `.gitignore` for unpublished content
4. **Backup Copies:** Keep backup versions with clear naming

## Troubleshooting

### Common Issues

**Math Not Rendering:**
- Verify `math: true` in front matter
- Check delimiter syntax (passthrough uses `\(` `\)` and `$$`)
- Ensure MathJax is loaded (check extend_head.html)

**Images Not Displaying:**
- Verify image file exists in post directory
- Check filename case sensitivity
- Ensure figure shortcode syntax is correct
- Verify path is relative (no leading `/`)

**Build Failures:**
- Check YAML front matter syntax
- Verify no unclosed shortcodes
- Ensure theme submodule is initialized
- Check for special characters in filenames

**Deployment Issues:**
- Verify push was to `main` branch
- Check GitHub Actions workflow status
- Ensure GITHUB_TOKEN permissions are correct
- Verify gh-pages branch exists

## Contact and Resources

- **Author:** Yue Shui
- **Email:** syhya925666582@gmail.com
- **GitHub:** https://github.com/syhya
- **LinkedIn:** https://www.linkedin.com/in/yue-shui/
- **Blog:** https://syhya.github.io/

## Additional Notes for AI Assistants

### Tone and Style

The blog maintains a professional, technical tone:
- Clear explanations of complex concepts
- Academic citation style
- Balanced between accessibility and technical depth
- Bilingual content should feel natural in both languages

### Content Scope

When assisting with content:
- Focus on LLM, AI, ML, and Deep Learning topics
- Maintain consistency with existing post topics
- Include practical applications and research insights
- Balance theory with implementation details

### Quality Checks

Before finalizing changes:
- [ ] Both language versions complete and synchronized
- [ ] All images present and properly referenced
- [ ] Math equations render correctly
- [ ] Citations formatted consistently
- [ ] Front matter complete and accurate
- [ ] Local build successful
- [ ] Commit message descriptive and clear

### Respect Existing Patterns

When making changes:
- Follow established naming conventions
- Maintain consistent directory structure
- Preserve existing shortcode patterns
- Keep configuration changes minimal
- Document significant modifications

---

**Last Updated:** 2025-11-27
**Version:** 1.0
**Maintained by:** AI Assistant (Claude)
