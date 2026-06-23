# al-folio v1 Migration Progress

Saved: 2026-06-13 18:05 CST

Current branch: `upgrade-al-folio-v1`

## Goal

Upgrade this personal homepage to the official al-folio v1 starter from `NewTemp/al-folio-main.zip`, while preserving Xin Zou's personal content:

- publications in `_bibliography/papers.bib`
- about page content in `_pages/about.md`
- CV information
- venue colors in `_data/venues.yml`
- publication customizations: `eqauthors` equal contribution marker and `highlight`

## Completed

- Unzipped official template to `/private/tmp/al-folio-upgrade/al-folio-main`.
- Replaced old tracked template files with the new al-folio v1 starter.
- Preserved/restored personal data:
  - `_bibliography/papers.bib`
  - `_pages/about.md`
  - `_pages/publications.md`
  - `_data/coauthors.yml`
  - `_data/venues.yml`
  - `_data/repositories.yml`
  - `assets/img/`
  - `assets/pdf/CV.pdf`
  - `assets/pdf/CV-Chinese.pdf`
- Configured v1 `_config.yml` for this site:
  - `first_name: Xin`
  - `last_name: Zou`
  - `url: https://ZouXinn.github.io`
  - empty `baseurl`
  - `theme: al_folio_core`
  - al-folio plugin stack enabled
  - Bootstrap compatibility enabled
  - scholar self-identification set to `Xin Zou`
  - publication badge settings adjusted
- Added `_data/socials.yml` with:
  - email
  - GitHub
  - X/Twitter
  - Google Scholar user id `CGtG970AAAAJ`
  - DBLP
- Migrated publication override to `_layouts/bib.liquid`.
  - Verified venue colors render.
  - Verified `Xin Zou*` / equal contribution renders.
  - Verified `highlight` renders, e.g. `Solve a COLT 2014 open problem`.
- Added `.al-folio-overrides.yml`.
- `NewTemp/` and `.DS_Store` were added to `.gitignore`.
- Restored official v1 feature pages:
  - `_pages/blog.md`
  - `_pages/books.md`
  - `_pages/news.md`
  - `_pages/profiles.md`
  - `_pages/projects.md`
  - `_pages/repositories.md`
  - `_pages/teaching.md`
  - `_pages/dropdown.md`
- Restored official v1 sample collections/assets:
  - `_posts/`
  - `_projects/`
  - `_books/`
  - `_teachings/`
  - `_news/`
  - `_pages/about_einstein.md`
  - `assets/jupyter/`
- Restored the official v1 CV interface:
  - Converted `_data/cv.yml` into RenderCV-style YAML under `cv:`.
  - Updated `_pages/cv.md` to `layout: cv` and `cv_format: rendercv`.
  - Removed the local `_layouts/cv.liquid` override so the `al_folio_cv` plugin renders `/cv/`.
  - Removed stale old `_includes/cv/*.html` files.
  - Left `.al-folio-overrides.yml` acknowledging only the intentional publication override `_layouts/bib.liquid`.
- Re-enabled restored demo/new-template features by keeping `_pages/about_einstein.md` and `assets/jupyter/` out of `_config.yml` `exclude`.
- Verified Jekyll build passes:
  - `docker compose exec jekyll bundle exec jekyll build --destination /tmp/_site-check`
- Ran upgrade audits:
  - `bundle exec al-folio upgrade audit --no-fail`: blocking `0`, non-blocking `8`.
  - `bundle exec al-folio upgrade overrides audit`: only `_layouts/bib.liquid`, acknowledged.
- Validated rendered pages via generated output:
  - `/cv/` contains RenderCV sections such as Contact Information, Education, Professional Services, Honors and Awards, Academic Interests, and CV Files.
  - `/blog/`, `/projects/`, `/repositories/`, `/books/`, and `/teaching/` are restored and build successfully.
  - `/publications/` still renders venue colors, `eqauthors` stars, and `highlight`.
- Restored the main navigation to the personal-site shape:
  - visible top nav: `About`, `Publications`, `CV`, `Blog`.
  - kept other v1 feature pages available by URL, but removed demo pages such as Projects, Repositories, Teaching, People, and the dropdown from the top navigation.
- Fixed the navbar rendering issue where the links could disappear inside the Bootstrap/Tailwind collapsed menu:
  - Added an intentional `_includes/header.liquid` override.
  - The header now renders `Xin Zou` plus `About`, `Publications`, `CV`, and `Blog` as always-visible links.
  - Rebuilt both `/srv/jekyll/_site` and `/tmp/_site`.
  - Acknowledged the header override in `.al-folio-overrides.yml`.

## Remaining Notes

- The current audit warnings are non-blocking and mostly come from intentional legacy compatibility or official sample posts:
  - jQuery use in `_includes/figure.html`.
  - jQuery use in the custom `_layouts/bib.liquid`.
  - sample post markup in official demo posts.
- The Docker dev server was previously running on `http://localhost:8080`.
- `NewTemp/al-folio-main.zip` is ignored and should not be committed.
- Do not commit yet unless the user asks.

## Notes

- Shell GitHub network was unreliable earlier, but Docker container networking worked for bundle install.
