# Health hidden set

200 unlabelled images for the final inference pass. Run your finalized six-agent system
over them and submit one 200-row CSV.

- `images/` — 200 ISIC JPEGs.
- `records.csv` — `image_id, age, sex, localization`. `dx`, `dx_type` and `lesion_id` are
  not present.
- Use the official `image_id` for every row of `health_submission.csv` (exactly 200 rows;
  missing, duplicate or unknown IDs reject the file).

Header-only template: `health_submission.csv` (download from the stage page).
