# Contributing to VeriNews

VeriNews is an academic/demo application. Contributions should preserve the current local Streamlit workflow and keep documentation aligned with the code that actually runs.

## Before Opening a Change

1. Read [README.md](README.md) and [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
2. Confirm that the change does not expose secrets, local databases, large datasets, or serialized artifacts.
3. Keep claims about models, accuracy, verification, and integrations precise and reproducible.
4. Run the application locally when the change affects the Streamlit flow.

## Documentation Changes

Documentation should distinguish clearly between:

- implemented runtime behavior;
- experimental or disconnected modules; and
- future ideas.

Avoid describing publisher reputation as proof of claim truth. Avoid presenting model confidence as calibrated certainty unless calibration evidence is included.

## Pull Requests

Include:

- a short summary of the change;
- the files affected;
- manual validation steps and results; and
- screenshots for visible UI changes.

Do not include API keys, `.env` files, `verinews.db`, raw datasets, or model binaries in commits.
