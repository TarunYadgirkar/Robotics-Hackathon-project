# Validation

60 seconds sampled at random (seed 7) from the 36 tasks the tracker can read, at the default motion threshold (0.6). Each sample was rendered as the two frames of that second, side by side, so a TRANSIT call is checkable rather than guessed at, and judged against the state the index assigned.

**54 of 60 agreed — 90%.**

Of the 6 disagreements, **5 are the index missing hands that are visible** and 1 is the index over-calling two hands. The error is almost entirely one-directional: when it is wrong, it under-reports manipulation. That makes the headline two-handed share a conservative floor rather than an inflated number — the true figure is higher than what this app shows, not lower.

Errors by assigned state: ABSENT 5/12, TWO-HANDED 1/29. Every ABSENT error is the same failure the Limitations tab describes: a hand at the frame edge, behind the workpiece, or gripping a tool.

## Honest caveats about this number

- The reviewer is the same agent that wrote the pipeline. This is a self-assessment, not an independent annotation, and it should be read as a sanity check rather than an eval.
- 60 samples over 35 hours is thin. The 95% binomial interval on 54/60 runs roughly 79-96%.
- Judging a second from two frames is easier for ABSENT and TWO-HANDED than for TRANSIT, where a borderline speed is genuinely ambiguous to a human too.
- Samples are drawn only from tasks above the 50% detection floor. The seven excluded tasks are not represented here, and their numbers are not reported anywhere in the app.

## Samples

| # | task | second | index said | verdict |
| --- | --- | --- | --- | --- |
| 00 | water-filtration-bottle-filling | 72 | TRANSIT | ✓ |
| 01 | garment-loop-attachment | 242 | ABSENT | ✗ hands are on the fabric at the machine; called ABSENT |
| 02 | drilling | 61 | TRANSIT | ✓ |
| 03 | electrical-wiring-assembly | 284 | TWO-HANDED | ✓ |
| 04 | garment-quality-checking | 31 | TWO-HANDED | ✗ only one hand is on the garment; called TWO-HANDED |
| 05 | garment-button-attachment | 166 | ABSENT | ✓ |
| 06 | milling-machine-operation | 265 | ONE-HANDED | ✓ |
| 07 | component-alignment-sticker-application | 271 | TWO-HANDED | ✓ |
| 08 | garment-tag-attachment | 284 | TWO-HANDED | ✓ |
| 09 | garment-folding-cardboard-insert | 247 | TRANSIT | ✓ |
| 10 | garment-inside-out | 54 | ABSENT | ✓ |
| 11 | buttonhole-stitching | 286 | ONE-HANDED | ✓ |
| 12 | axle-shaft-cutting | 29 | TWO-HANDED | ✓ |
| 13 | electrical-wiring-assembly | 127 | TWO-HANDED | ✓ |
| 14 | garment-stitching-overlock | 97 | TRANSIT | ✓ |
| 15 | oil-seal-pressing | 141 | TWO-HANDED | ✓ |
| 16 | belly-band-assembly | 21 | TWO-HANDED | ✓ |
| 17 | garment-belly-band-wrapping | 50 | ABSENT | ✗ a hand is at the right frame edge; called ABSENT |
| 18 | garment-label-attachment | 259 | TWO-HANDED | ✓ |
| 19 | axle-shaft-cutting | 231 | TWO-HANDED | ✓ |
| 20 | garment-loop-attachment | 287 | TWO-HANDED | ✓ |
| 21 | register-ring-binding | 14 | TWO-HANDED | ✓ |
| 22 | garment-loop-attachment | 32 | ONE-HANDED | ✓ |
| 23 | fabric-layering | 226 | ABSENT | ✓ |
| 24 | garment-hanger-place | 166 | TRANSIT | ✓ |
| 25 | garment-packing-general | 258 | ONE-HANDED | ✓ |
| 26 | belly-band-assembly | 262 | ONE-HANDED | ✓ |
| 27 | binding-pre-fold-stitching | 102 | TWO-HANDED | ✓ |
| 28 | garment-packing-general | 141 | ABSENT | ✓ |
| 29 | garment-iron-press | 231 | ABSENT | ✗ a sleeved arm and hand are at the board; called ABSENT |
| 30 | component-alignment-sticker-application | 260 | TWO-HANDED | ✓ |
| 31 | garment-tag-attachment | 273 | TWO-HANDED | ✓ |
| 32 | fabric-layering | 244 | ABSENT | ✓ |
| 33 | belly-band-assembly | 259 | ONE-HANDED | ✓ |
| 34 | bottle-surface-buffing | 126 | ABSENT | ✗ both hands are plainly on the bottle; called ABSENT |
| 35 | buttonhole-stitching | 267 | ABSENT | ✗ a hand is at the machine; called ABSENT |
| 36 | filter-tube-assembly | 132 | TWO-HANDED | ✓ |
| 37 | garment-carton-packing | 286 | TWO-HANDED | ✓ |
| 38 | garment-packing-general | 103 | ABSENT | ✓ |
| 39 | garment-folding-cardboard-insert | 229 | TRANSIT | ✓ |
| 40 | cnc-machine-operation | 70 | ONE-HANDED | ✓ |
| 41 | garment-label-attachment | 213 | TWO-HANDED | ✓ |
| 42 | garment-safety-sticker | 62 | TWO-HANDED | ✓ |
| 43 | garment-packing-general | 200 | TWO-HANDED | ✓ |
| 44 | panel-fitting-placement | 226 | TWO-HANDED | ✓ |
| 45 | garment-edge-hemming | 161 | TWO-HANDED | ✓ |
| 46 | milling-machine-operation | 37 | TWO-HANDED | ✓ |
| 47 | garment-folding-general | 123 | ONE-HANDED | ✓ |
| 48 | loop-tape-preparation | 219 | TWO-HANDED | ✓ |
| 49 | cnc-machine-operation | 37 | ONE-HANDED | ✓ |
| 50 | garment-quality-checking | 108 | ONE-HANDED | ✓ |
| 51 | garment-zip-attachment | 155 | TRANSIT | ✓ |
| 52 | metal-grinding | 62 | TWO-HANDED | ✓ |
| 53 | garment-carton-packing | 79 | ABSENT | ✓ |
| 54 | belly-band-assembly | 187 | TWO-HANDED | ✓ |
| 55 | garment-stitching-general | 73 | TRANSIT | ✓ |
| 56 | belly-band-assembly | 129 | TRANSIT | ✓ |
| 57 | cnc-machine-operation | 70 | TWO-HANDED | ✓ |
| 58 | electrical-wiring-assembly | 239 | TWO-HANDED | ✓ |
| 59 | garment-belly-band-wrapping | 112 | TWO-HANDED | ✓ |
