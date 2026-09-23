# Architecture - FAST Temporal Subtitle Detector V1

## Goal

Detect hard-coded Chinese subtitles in video, draw one tight line-level
bounding box for every visible subtitle line, and keep geometry stable over
the complete lifetime of a subtitle.

## Production pipeline

1. Bottom ROI inference
   - Default ROI is the lower 45% of the frame.
   - FAST-B-736 produces scene-text components.

2. Two-level detection
   - High-confidence components form primary observations.
   - Lower-confidence components are retained only when temporal or geometry
     evidence supports them.

3. Line grouping and tracking
   - Components are grouped into subtitle lines.
   - Temporal association uses overlap, center distance and lifecycle rules.

4. Content-aware lifecycle segmentation
   - Content-aware logic separates real subtitle changes from detector noise.
   - Weak and single-character subtitle recovery is handled separately.

5. V1 logical subtitle stitching
   - Adjacent fragments with the same content are merged.
   - OCR similarity, visual-mask similarity, temporal adjacency and geometry
     compatibility are considered together.
   - Short cross-fade fragments are absorbed into a neighboring episode.

6. Static geometry lock
   - The evaluation video uses screen-static hard subtitles.
   - Temporal evidence becomes one canonical box per logical subtitle line.
   - The same geometry is reused for the full lifetime of that line.

7. Temporal glyph tightening
   - Persistent glyph/outline evidence removes excess background.
   - OCR preservation guards reject crops that lose subtitle content.

8. Multiline handling
   - Each subtitle line keeps an independent box.
   - Sibling widths are never forced to match.
   - Center/separation constraints prevent nested or overlapping boxes.

9. Single-line height consistency
   - Local font-height priors correct only abnormally tall boxes.
   - Normal-height and multiline boxes remain unchanged.

10. Output
    - MP4 with drawn boxes.
    - JSON with frame, timestamp, track/subtitle/line IDs and bbox geometry.
    - Original audio is remuxed with ffmpeg when available.

## Why not frame-by-frame smoothing?

Hard subtitles provide a stronger temporal prior than ordinary moving
objects: the same subtitle normally does not move on screen. V1 therefore
models a logical subtitle episode first and assigns canonical geometry to
that episode instead of continuously smoothing noisy per-frame boxes.
