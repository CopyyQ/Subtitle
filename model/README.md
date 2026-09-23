# Production model

This project uses one text detector only: FAST-B-736.

Run:

    bash scripts/setup_fast.sh

The script provisions the pinned FAST repository at model/FAST and verifies
the production checkpoint SHA-256. The model repository and checkpoint are
runtime dependencies and are intentionally not committed here.
