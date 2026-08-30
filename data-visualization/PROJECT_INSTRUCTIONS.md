# Factory POV Video to Navigable 3D

## Codex implementation brief

This document is the source of truth for building a data-exploration site for an existing library of first-person factory videos. The source videos are already hosted in Amazon S3. The system must inventory those videos, preprocess them in batch, and produce browser-navigable 3D Gaussian-splat environments and related analytical representations on AWS. Later phases add video overlays, hand and body joints, tools, movement paths, semantic search, and dynamic 3D hands.

The first release must prove one narrow outcome:

> Select an existing S3-hosted POV video and immediately explore its precomputed, browser-viewable 3D reconstruction of the static workstation.

Do not begin dynamic or 4D hand reconstruction until the static environment pipeline meets the acceptance criteria in this document.

### Existing-data and precomputation requirements

- Treat the current S3 video collection as the source dataset; do not make end-user upload the primary product workflow.
- Discover and register existing videos from configured private S3 prefixes without copying or re-uploading them unnecessarily.
- Run reconstruction and analytical extraction as an offline batch process across the catalog.
- Precompute all feasible 3D data, previews, camera poses, quality reports, masks, landmarks, paths, and browser-optimized representations before exposing a video as ready on the site.
- The exploration experience must load existing artifacts. It must not start GPU reconstruction in response to a normal page view or require a user to wait for training.
- Track processing coverage and status for every source video, including accepted, needs-review, rejected, failed, and not-yet-processed states.
- When a source video changes, use its S3 object key, version ID when available, and ETag to determine which derived artifacts need to be regenerated.

### Local administrative upload pipeline

The local exploration site also provides a drag-and-drop administrative ingestion path. An uploaded video must be stored in the configured S3 bucket before derived artifacts are published. The queued local worker must then reproduce the sample workflow for the entire uploaded video:

- preserve the source duration, dimensions, and frame rate in the browser video;
- run full-length hand detection, motion-based hand identity tracking, and short-gap interpolation;
- run WiLoR MANO inference for the full video at the video's actual frame rate;
- render the WiLoR overlay at the same duration and frame rate as the source;
- generate the interactive MANO data, relative-depth 3D video surface, and full-length depth heat-map video;
- upload every final artifact and manifest under a unique `derived/{video_id}/` S3 prefix;
- expose queued, active, completed, and failed stage status in the local site.

This upload control is an administrative addition; the existing S3 catalog remains the primary dataset.

---

## 1. Product objective

Build a private data-exploration web application where a user can:

1. Browse and search the catalog of factory POV videos already stored in S3.
2. Filter videos by processing status, quality, location, workstation, time, and available representations.
3. Open a video and play the original footage through secure delivery.
4. Inspect its processing status, provenance, and quality report.
5. Enable precomputed analytical overlays such as arm joints, hand joints, tools, and movement paths.
6. Switch at the same timestamp into a precomputed navigable 3D reconstruction.
7. Compare and explore videos and reconstructed environments without triggering reconstruction during the interactive session.

Phase 1 covers S3 catalog ingestion, offline static reconstruction, quality reporting, basic catalog browsing, and basic 3D navigation. Uploading new videos may be added as an administrative ingestion path, but it is not the primary user experience.

---

## 2. Important technical constraint

Standard 3D Gaussian Splatting assumes a mostly static scene. Factory POV footage contains moving hands, tools, parts, workers, and machinery. If those pixels are used directly during static reconstruction, they can become blurry streaks, duplicated objects, or floating geometry.

For Phase 1:

- Reconstruct the workstation, machinery, walls, floor, fixtures, and other static content.
- Detect and mask hands, arms, tools, workers, and obviously moving parts before training.
- Preserve the masks and camera poses for later dynamic reconstruction.
- Mark a scene as unsuitable when the video lacks sufficient camera translation, overlap, sharp frames, or unobstructed background coverage.

The raw video may not contain enough information to reconstruct surfaces never seen by the camera. The viewer must not imply that hallucinated or missing regions are measured ground truth.

---

## 3. Chosen technology stack

### Reconstruction worker

- Python 3.10 or newer, pinned to a version compatible with all CUDA dependencies
- FFmpeg for video inspection and frame extraction
- CUDA-enabled COLMAP for camera poses and sparse structure
- Nerfstudio `splatfacto` for Gaussian-splat training
- Nerfstudio/gsplat for Gaussian rendering and export
- SAM 2.1 Hiera Base+ for tracking masks through video
- MediaPipe Hand Landmarker as an initial source of hand bounding-box prompts
- Depth Anything V2 Small as an optional depth prior, not a required first-pass dependency
- AWS SDK for Python (`boto3`)

### AWS infrastructure

- Amazon S3 for raw videos, intermediate artifacts, and outputs
- Amazon ECR for reconstruction containers
- AWS Batch with EC2 GPU compute environments
- EC2 `g6e.2xlarge` as the initial GPU instance type
- AWS Step Functions for orchestration
- Amazon EventBridge for source-object discovery and change events
- AWS Lambda for lightweight validation and workflow initiation only
- Amazon DynamoDB for job and video metadata
- Amazon CloudWatch for logs, metrics, alarms, and dashboards
- Amazon CloudFront with Origin Access Control for private delivery
- AWS KMS for encryption keys
- Amazon Cognito, API Gateway, and Lambda for the application API in a later phase
- Amazon OpenSearch Service for semantic video and clip search in a later phase

### Web application

- React
- TypeScript
- Vite
- Three.js
- A maintained Three.js-compatible Gaussian-splat renderer wrapped behind an internal adapter
- Native HTML video playback for the original footage

### Infrastructure as code

- AWS CDK v2 in TypeScript
- Separate `dev` and `prod` stacks
- All resource names derived from project name, environment, account, and Region

---

## 4. What is and is not a model

| Component | Role | Model type |
|---|---|---|
| FFmpeg | Decode and sample frames | Not ML |
| COLMAP | Camera poses and sparse 3D structure | Geometric SfM pipeline, not a neural model |
| SAM 2.1 Hiera Base+ | Track masks for hands, tools, workers, and moving objects | Video segmentation foundation model |
| Splatfacto | Optimize the static 3D Gaussian scene | 3D Gaussian Splatting implementation |
| gsplat | CUDA Gaussian rasterization backend | Rendering/optimization library |
| Depth Anything V2 Small | Optional per-frame depth estimates | Monocular depth model |
| MediaPipe Hand Landmarker | Hand landmarks and initial prompt boxes | Hand landmark model |
| MediaPipe Pose Landmarker | Later arm/body overlay | Pose landmark model |

Do not use Amazon Bedrock as the reconstruction engine. The core reconstruction is geometric optimization running in a custom GPU container.

Official references:

- [Nerfstudio custom video processing](https://docs.nerf.studio/quickstart/custom_dataset.html)
- [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [COLMAP Structure-from-Motion](https://colmap.github.io/tutorial.html)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions)
- [AWS Batch GPU jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- [Step Functions and AWS Batch](https://docs.aws.amazon.com/step-functions/latest/dg/connect-batch.html)

---

## 5. Repository structure

Codex should create the project with this structure:

```text
factory-video-3d/
├── README.md
├── PROJECT_INSTRUCTIONS.md
├── Makefile
├── .gitignore
├── .env.example
├── docs/
│   ├── architecture.md
│   ├── capture-guidelines.md
│   ├── operations.md
│   └── quality-metrics.md
├── infra/
│   ├── package.json
│   ├── tsconfig.json
│   ├── cdk.json
│   ├── bin/
│   │   └── app.ts
│   ├── lib/
│   │   ├── storage-stack.ts
│   │   ├── batch-stack.ts
│   │   ├── pipeline-stack.ts
│   │   ├── delivery-stack.ts
│   │   └── observability-stack.ts
│   └── test/
├── worker/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/factory3d/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── storage.py
│   │   ├── probe.py
│   │   ├── frames.py
│   │   ├── masks.py
│   │   ├── poses.py
│   │   ├── train.py
│   │   ├── export.py
│   │   ├── quality.py
│   │   └── manifest.py
│   ├── scripts/
│   │   ├── entrypoint.sh
│   │   └── smoke_test.sh
│   └── tests/
├── api/
│   ├── pyproject.toml
│   ├── src/
│   └── tests/
├── web/
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── viewers/
│   │   │   ├── VideoViewer.tsx
│   │   │   ├── SplatViewer.tsx
│   │   │   └── SplatRendererAdapter.ts
│   │   └── api/
│   └── tests/
└── fixtures/
    └── README.md
```

Never commit customer factory footage, model checkpoints, generated frames, splats, credentials, or AWS account identifiers to source control.

---

## 6. S3 layout

The source videos already exist in private S3 storage. Make the source bucket and one or more source prefixes configurable. The pipeline must be able to read from the existing source location and write derived artifacts to a separate private bucket or configured output prefix.

Use these logical prefixes for registered source references and generated data:

```text
s3://{bucket}/raw/{video_id}/source.mp4
s3://{bucket}/jobs/{video_id}/{job_id}/request.json
s3://{bucket}/intermediate/{video_id}/{job_id}/frames/
s3://{bucket}/intermediate/{video_id}/{job_id}/masks/
s3://{bucket}/intermediate/{video_id}/{job_id}/colmap/
s3://{bucket}/intermediate/{video_id}/{job_id}/checkpoints/
s3://{bucket}/scenes/{video_id}/{scene_version}/scene.ply
s3://{bucket}/scenes/{video_id}/{scene_version}/scene.web
s3://{bucket}/scenes/{video_id}/{scene_version}/cameras.json
s3://{bucket}/scenes/{video_id}/{scene_version}/preview.jpg
s3://{bucket}/scenes/{video_id}/{scene_version}/quality.json
s3://{bucket}/scenes/{video_id}/{scene_version}/manifest.json
```

The example `raw/` key is not a requirement to relocate existing videos. Store each video's actual source bucket, object key, version ID when available, and ETag in metadata. Generate stable `video_id` values during catalog registration and avoid duplicating source objects solely to fit the example layout.

`scene.ply` is the canonical export. `scene.web` represents whichever compressed browser format is chosen during implementation. Do not delete the canonical export when generating browser-optimized derivatives.

Add lifecycle rules:

- Raw videos: retain according to product policy.
- Intermediate frames, masks, and temporary COLMAP data: expire after 7 days by default.
- Checkpoints: expire after 7 days unless a job is marked for investigation.
- Final scenes and manifests: retain.
- Failed-job artifacts: retain for 14 days in development and according to policy in production.

---

## 7. Job request and output contracts

### Request

```json
{
  "schema_version": "1.0",
  "job_id": "uuid",
  "video_id": "uuid",
  "input_s3_uri": "s3://bucket/raw/video-id/source.mp4",
  "output_s3_prefix": "s3://bucket/scenes/video-id/v1/",
  "settings": {
    "frame_target": 150,
    "mask_dynamic_objects": true,
    "use_depth_prior": false,
    "trainer": "splatfacto",
    "quality_profile": "preview"
  }
}
```

### Manifest

```json
{
  "schema_version": "1.0",
  "video_id": "uuid",
  "job_id": "uuid",
  "scene_version": "v1",
  "status": "accepted",
  "created_at": "ISO-8601 timestamp",
  "source": {
    "duration_seconds": 30.0,
    "width": 1920,
    "height": 1080,
    "fps": 30.0
  },
  "processing": {
    "frames_extracted": 150,
    "frames_registered": 132,
    "registration_ratio": 0.88,
    "dynamic_masks_applied": true,
    "depth_prior_applied": false,
    "trainer": "splatfacto"
  },
  "artifacts": {
    "canonical_splat": "scenes/video-id/v1/scene.ply",
    "web_splat": "scenes/video-id/v1/scene.web",
    "cameras": "scenes/video-id/v1/cameras.json",
    "preview": "scenes/video-id/v1/preview.jpg",
    "quality": "scenes/video-id/v1/quality.json"
  }
}
```

Version every external JSON contract. Parsers must reject unsupported major versions and tolerate additive fields within a supported major version.

---

## 8. Worker pipeline

The worker must implement these stages as explicit, restartable steps. Each stage writes structured logs and a stage-completion marker.

### Stage A: download and validate

1. Parse and validate the job request.
2. Download the source video from S3 into an isolated working directory.
3. Use `ffprobe` to collect duration, resolution, frame rate, codec, rotation metadata, and audio presence.
4. Reject corrupt videos and unsupported codecs with a machine-readable reason.
5. Normalize rotation metadata before frame processing.
6. Enforce configurable limits for duration, resolution, and input size.

### Stage B: suitability analysis

Calculate inexpensive signals before spending significant GPU time:

- Blur distribution
- Exposure clipping
- Duplicate-frame rate
- Estimated camera motion
- Optical-flow magnitude
- Percentage of frames dominated by hands or foreground objects
- Initial feature count and feature-match connectivity

Return `rejected` before training when there is clearly insufficient motion, overlap, texture, or usable background coverage.

### Stage C: frame selection

1. Sample frames across the usable time range.
2. Prefer sharp frames with meaningful viewpoint changes.
3. Avoid keeping many nearly identical consecutive frames.
4. Preserve timestamps for every selected frame.
5. Write `frames.json` mapping image filenames to source timestamps.

Start with 100–300 selected frames per scene. Make this configurable and tune it using pilot results rather than assuming more frames always improve reconstruction.

### Stage D: dynamic-object masking

1. Run MediaPipe Hand Landmarker on sampled frames.
2. Convert detected hands into padded bounding-box prompts.
3. Initialize SAM 2.1 video objects from those prompts.
4. Track hand and arm masks through the selected sequence.
5. Add prompts for tools, workers, and moving machine components when automatically detected or manually supplied.
6. Dilate masks slightly to avoid training on motion boundaries.
7. Save one binary mask per selected image.
8. Record mask coverage and confidence statistics.

When automatic prompting fails:

- Mark the job `needs_review`.
- Preserve generated frames.
- Support a later review interface where an operator can add a point, box, or mask prompt and resume from the masking stage.

### Stage E: camera poses and sparse geometry

Run Nerfstudio preprocessing, which uses FFmpeg and COLMAP:

```bash
ns-process-data video \
  --data /work/source.mp4 \
  --output-dir /work/processed \
  --num-frames-target 150
```

The production implementation may perform custom frame selection before COLMAP. In that case, use the Nerfstudio images workflow or generate the equivalent processed dataset while preserving timestamps and masks.

Capture:

- Number of input frames
- Number of registered frames
- Registration ratio
- Mean and percentile reprojection errors
- Camera path extent
- Sparse point count
- Failed or disconnected image groups

Fail cleanly when COLMAP cannot produce one sufficiently connected reconstruction.

### Stage F: Gaussian training

Start with standard Splatfacto:

```bash
ns-train splatfacto --data /work/processed
```

Requirements:

- Run without exposing the training viewer publicly.
- Write outputs and checkpoints under the job working directory.
- Record the resolved Nerfstudio, gsplat, PyTorch, CUDA, and driver versions.
- Save the final configuration file.
- Use deterministic seeds where supported.
- Add a configurable wall-clock timeout.
- Upload periodic checkpoints when running on Spot capacity.

Do not enable `splatfacto-big` until the default pipeline is stable and measured.

### Stage G: export

Export the trained Gaussian model:

```bash
ns-export gaussian-splat \
  --load-config /work/outputs/.../config.yml \
  --output-dir /work/export
```

Generate:

- Canonical Gaussian `.ply`
- Browser-optimized compressed splat
- Camera path JSON
- Preview render from one or more viewpoints
- Processing manifest
- Quality report

Verify that the exported scene opens in the selected browser renderer before marking the job accepted.

### Stage H: upload and finalize

1. Upload artifacts to the versioned S3 scene prefix.
2. Update the DynamoDB job record atomically.
3. Emit a completion event.
4. Delete local scratch data.
5. Never mark a job successful unless the manifest, canonical splat, web splat, preview, and quality report all exist.

---

## 9. Quality report

`quality.json` must include raw metrics and a decision. Do not store only a single opaque score.

Suggested fields:

```json
{
  "decision": "accepted",
  "reasons": [],
  "blur": {
    "median": 0.0,
    "p10": 0.0
  },
  "camera": {
    "registered_ratio": 0.88,
    "reprojection_error_mean": 0.0,
    "path_extent": 0.0,
    "connected_components": 1
  },
  "masking": {
    "mean_dynamic_coverage": 0.21,
    "max_dynamic_coverage": 0.58,
    "low_confidence_frames": 4
  },
  "scene": {
    "gaussian_count": 0,
    "artifact_size_bytes": 0
  },
  "viewer_check": {
    "loaded": true,
    "preview_rendered": true
  }
}
```

Initial decision guidance:

- `accepted`: suitable for the viewer without manual intervention.
- `needs_review`: usable frames and poses exist, but masking or visual quality needs review.
- `rejected`: insufficient data or unrecoverable processing failure.

Treat thresholds as configuration, not constants scattered through code. Finalize thresholds after processing at least 20 representative pilot videos.

---

## 10. AWS Batch design

Create two job queues:

### CPU/preflight queue

For validation, probing, and inexpensive suitability analysis. Use general-purpose EC2 or Fargate only when the dependency set is compatible.

### GPU reconstruction queue

Initial configuration:

- Managed EC2 compute environment
- `g6e.2xlarge`
- One GPU requested per job
- Minimum vCPUs: 0
- Development maximum: equivalent to 2 concurrent GPU jobs
- Production maximum: configurable
- On-Demand for the initial pilot
- Spot environment added after checkpoint/retry behavior is proven
- Encrypted `gp3` scratch volume sized for video, frames, model checkpoints, and exports
- GPU-capable ECS-optimized AMI or a validated custom AMI

Use separate On-Demand and Spot compute environments attached to the same queue, with On-Demand available as fallback according to configured order.

Every Batch job must set:

- Job timeout
- Retry strategy
- CloudWatch log group
- CPU, memory, and GPU requirements
- Read-only container root filesystem where practical
- Non-root process where compatible with CUDA dependencies
- Environment variables that contain identifiers and S3 paths, never long-lived credentials

---

## 11. Step Functions workflow

Implement this state machine:

```text
Validate request
    ↓
Create job record
    ↓
Submit preflight Batch job and wait
    ↓
Preflight acceptable?
    ├── no → mark rejected or needs_review
    └── yes
         ↓
Submit GPU reconstruction job and wait
         ↓
Verify required artifacts
         ↓
Update job/video records
         ↓
Publish completion event
```

Use the native synchronous Batch integration:

```text
arn:aws:states:::batch:submitJob.sync
```

Do not poll Batch from Lambda.

Handle:

- Retryable infrastructure failures
- Spot interruption
- Invalid input
- COLMAP registration failure
- GPU out-of-memory
- Training timeout
- Missing artifacts
- Manual-review outcomes

Include idempotency. Replaying the same workflow execution must not overwrite an accepted scene version or create inconsistent metadata.

---

## 12. DynamoDB records

Use one table initially with a composite key:

```text
PK = VIDEO#{video_id}
SK = METADATA

PK = VIDEO#{video_id}
SK = JOB#{job_id}

PK = VIDEO#{video_id}
SK = SCENE#{scene_version}
```

Store:

- Current processing state
- Created and updated timestamps
- Source object key and ETag
- Workflow execution ARN
- Batch job IDs
- Scene versions
- Quality decision
- Artifact keys
- Error category and sanitized error message

Use conditional writes for state transitions. Do not expose raw internal stack traces through the application API.

---

## 13. Browser viewer milestone

The initial viewer must:

1. Provide a catalog view of registered S3 videos and clearly show which representations are ready.
2. Load scene metadata from a precomputed manifest.
3. Request a short-lived signed URL or CloudFront signed URL.
4. Stream or load the browser-optimized splat without invoking a reconstruction job.
5. Orbit, pan, and zoom.
6. Provide a reset-camera control.
7. Display loading progress and a clear error state.
8. Work on current desktop Chrome, Edge, and Safari.
9. Avoid loading the original `.ply` when a smaller browser artifact exists.
10. Record load time and rendering failures.

Wrap the renderer behind `SplatRendererAdapter` so the application is not tightly coupled to one third-party viewer library.

Phase 1 needs a basic catalog UI sufficient to browse the existing dataset and open ready videos. It does not need advanced semantic search, joint overlays, or synchronized video/3D playback.

---

## 14. Security requirements

- Keep all S3 buckets private.
- Use CloudFront Origin Access Control.
- Use signed delivery URLs for scene artifacts.
- Encrypt S3, DynamoDB, EBS, logs, and secrets with KMS-managed keys according to environment policy.
- Block public access at account and bucket levels.
- Use least-privilege IAM roles for Step Functions, Batch instances, Batch jobs, Lambda, and CloudFront.
- Give each worker access only to required prefixes where practical.
- Store no AWS access keys in code, images, environment files, or CI variables when workload roles are available.
- Use ECR image scanning.
- Pin container base images and Python/Node dependencies.
- Produce a software bill of materials for production images.
- Do not log signed URLs, credentials, complete customer object paths, or video frames.
- Define retention and deletion behavior before processing real customer footage.

Factory footage can expose employees, processes, equipment, and proprietary operations. Treat it as sensitive business data.

---

## 15. Observability

### Structured logs

Every log record should include:

- `job_id`
- `video_id`
- `scene_version`
- `stage`
- `event`
- `elapsed_seconds`
- `severity`

### Metrics

Publish:

- Jobs submitted, running, accepted, rejected, and failed
- Queue wait time
- Processing time by stage
- GPU job duration
- Frame registration ratio
- Mask coverage
- Output size
- Viewer load time
- Failure count by category

### Alarms

Create alarms for:

- Repeated workflow failures
- GPU jobs stuck in `RUNNABLE`
- High rejection rate
- No successful jobs over an expected interval
- Unexpected spend or GPU concurrency
- S3 artifact verification failures

---

## 16. Cost controls

- Scale Batch compute environments to zero when idle.
- Limit development to two concurrent GPU jobs.
- Add Spot only after checkpointing and idempotent retries work.
- Use AWS Batch `SPOT_PRICE_CAPACITY_OPTIMIZED` for the Spot environment.
- Set job timeouts so failed COLMAP or training jobs cannot run indefinitely.
- Run preflight checks before allocating a GPU.
- Use S3 lifecycle policies for intermediates and checkpoints.
- Retain one canonical splat and regenerate derived formats when necessary.
- Tag every resource with project, environment, owner, and cost-center fields.
- Process 20 pilot videos and calculate median and percentile GPU-minutes per accepted scene before setting production capacity.

Do not use FSx for Lustre in the MVP. Add it only if measured data-loading behavior shows that independent S3-to-local job staging is a bottleneck.

---

## 17. Testing strategy

### Unit tests

- Request and manifest schema validation
- S3 URI parsing
- State transitions
- Frame timestamp mapping
- Quality decision logic
- Error categorization
- Idempotency helpers

### Container smoke test

- GPU visible through CUDA
- FFmpeg and ffprobe available
- COLMAP reports CUDA support
- Nerfstudio imports
- SAM 2.1 checkpoint loads
- Splatfacto performs a minimal training invocation
- Export command produces a readable artifact

### Integration test

Use a small, legally distributable fixture capture:

1. Place the fixture video in a test S3 source prefix.
2. Run catalog discovery and verify that the source object is registered.
3. Start the offline processing workflow.
4. Wait for Batch completion.
5. Verify DynamoDB transitions.
6. Verify required S3 artifacts.
7. Load the web splat in an automated browser test without starting another processing job.
8. Save a screenshot for visual regression review.

### Pilot test

Use at least 20 representative factory POV videos spanning:

- Different stations
- Different lighting
- Gloves and bare hands
- Reflective metal
- Fast and slow head motion
- Heavy and light occlusion
- Different resolutions and frame rates

Record success rate and failure reasons. Do not tune only against one easy video.

---

## 18. Phase plan and definitions of done

### Phase 0: capture assessment

Deliverables:

- Capture guidelines
- Inventory of the existing S3 video prefixes and object metadata
- A representative sample drawn from the existing catalog
- Automated probe reports for a representative subset
- Decision on how much existing footage contains adequate parallax

Done when the existing catalog is registered and at least one sample is judged reconstructable.

### Phase 1A: local/container proof

Deliverables:

- Reproducible GPU container
- Raw video to processed Nerfstudio dataset
- COLMAP camera poses
- Splatfacto checkpoint
- Exported `.ply`
- Local viewer validation

Done when one sample produces a recognizable navigable static workstation.

### Phase 1B: AWS reconstruction pipeline

Deliverables:

- CDK stacks
- Private S3 storage
- ECR repository
- Batch queues and compute environments
- Step Functions workflow
- DynamoDB metadata
- CloudWatch observability
- Existing S3 catalog discovery and registration
- One existing S3 video processed end to end without re-upload

Done when registered S3 videos can be processed in offline batches to produce versioned, verified scene artifacts without manual server access.

### Phase 1C: browser viewer

Deliverables:

- Private video catalog and data-exploration viewer
- Signed artifact delivery
- Orbit, pan, zoom, reset, loading, and errors
- Automated scene-loading test

Done when an authorized user can browse existing videos and immediately open a precomputed reconstructed scene in a browser without a running GPU server.

### Phase 2: static-scene quality improvements

Deliverables:

- SAM 2.1 dynamic masks
- Manual mask-review fallback
- Optional depth prior
- Quality reports and rejection reasons
- Pilot results for 20 videos

Done when failures are detected early and accepted scenes meet agreed visual criteria.

### Phase 3: synchronized video overlays

Deliverables:

- Video player
- Shared timeline
- Hand and pose landmarks
- Tool boxes and movement paths
- Toggleable overlays

Done when overlays remain synchronized with the source video and are stored as timestamped data rather than burned into the video.

### Phase 4: dynamic 3D hands and tools

Do not select a dynamic Gaussian method until Phase 2 data has been reviewed. Evaluate dynamic/4D Gaussian approaches against the actual capture setup. Monocular footage may not provide sufficient geometry for reliable free-viewpoint hands.

Done only when moving objects remain temporally stable from nearby novel viewpoints and limitations are clearly represented.

### Phase 5: advanced catalog and semantic explorer

Deliverables:

- Video and clip embeddings
- OpenSearch vector index
- Text and similarity search
- 2D or 3D semantic atlas
- Selection into the dual-mode viewer

Done when search ranking uses original high-dimensional embeddings; any 2D/3D projection is used only for browsing.

---

## 19. Codex execution order

Codex should implement in this order:

1. Confirm the repository location and ensure unrelated files are untouched.
2. Create the project structure and basic documentation.
3. Add JSON schemas for catalog records, requests, manifests, and quality reports.
4. Implement S3 catalog discovery, stable video registration, and change detection.
5. Implement and test video probing and frame selection locally.
6. Build the worker container and its smoke test.
7. Validate Nerfstudio video processing with a fixture capture.
8. Validate Splatfacto training and export on one GPU.
9. Add S3 input/output handling and idempotent stage markers.
10. Implement CDK storage and ECR stacks.
11. Implement the AWS Batch GPU environment and job definition.
12. Implement Step Functions orchestration and DynamoDB state transitions.
13. Add CloudWatch logs, metrics, alarms, and cost controls.
14. Run one existing S3 video through end-to-end AWS reconstruction.
15. Add batch scheduling and coverage tracking for the existing catalog.
16. Build the minimal catalog and browser splat viewer.
17. Add signed video and artifact delivery.
18. Add masking and manual-review support.
19. Precompute all feasible Phase 1 representations for the pilot catalog.
20. Run the 20-video pilot and tune thresholds.
21. Begin overlay and dynamic-scene work only after the static pipeline is accepted.

For every step, Codex must:

- Preserve unrelated workspace files.
- Use infrastructure as code.
- Add automated tests proportional to risk.
- Verify commands against pinned dependency versions.
- Keep credentials and customer video out of source control.
- Record assumptions in the relevant documentation.
- Stop and request user input before creating paid AWS resources if deployment authority has not been explicitly granted.

---

## 20. Inputs required before implementation can run

The project can be scaffolded without these inputs, but an end-to-end reconstruction requires:

1. Read access to the private S3 bucket and prefixes containing the current videos.
2. The source objects' key pattern and any available catalog metadata.
3. One representative 15–60 second POV factory video selected from that collection.
4. AWS account and deployment authorization.
5. Target AWS Region with G6e availability and sufficient quota.
6. Development GPU concurrency limit.
7. Data-retention and employee-privacy requirements.
8. Confirmation that the selected model licenses are acceptable for the intended commercial use.
9. Optional domain and authentication requirements for the viewer.

The first sample should have slow head movement, visible static surroundings, limited blur, and repeated views of the workstation. Avoid beginning with the most difficult footage.

---

## 21. Phase 1 acceptance criteria

Phase 1 is complete when all of the following are true:

- Existing videos can be discovered and registered from the configured private S3 prefixes without requiring re-upload.
- Every registered video has a visible processing state and source-version provenance.
- The workflow starts without manual server access.
- Preflight rejects clearly unsuitable footage before long GPU training.
- COLMAP produces one connected camera solution for an acceptable sample.
- Splatfacto completes and exports a canonical `.ply`.
- The pipeline creates a browser-optimized scene artifact.
- A manifest and quality report are present and schema-valid.
- DynamoDB shows a consistent terminal state.
- Ready videos have their required 3D and browser representations precomputed before they appear as explorable on the site.
- Opening a ready video or scene does not launch a GPU reconstruction job.
- An authorized user can browse the catalog and select an existing video.
- An authorized browser can orbit, pan, and zoom through the scene.
- The GPU compute environment returns to zero capacity after processing.
- Logs contain no credentials, signed URLs, or raw frames.
- Re-running the same request is idempotent.
- Failure cases produce actionable, machine-readable reasons.

Phase 1 is not complete merely because a training command exits successfully. The exported scene must be recognizable, navigable, securely delivered, and operationally repeatable.
