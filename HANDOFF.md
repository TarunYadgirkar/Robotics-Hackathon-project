# YAM arm — collision-avoidance work, handoff

Written for whoever picks this up next. Everything below is either measured on
this hardware or flagged as unverified. **The arm has never executed a planned
path.** Nothing has moved except two hand-bounded nudges early on.

---

## 1. The setup

- **Arm**: i2rt YAM (6 DoF + gripper), 7 Damiao servos on CAN at 1 Mbit/s.
  Joints 1–3 are DM4340, joints 4–6 and the gripper DM4310.
- **Link**: CANable 2.0 USB-to-CAN, macOS, driven through `python-can`'s
  `gs_usb` backend. There is no SocketCAN on macOS, so the vendor SDK's normal
  transport does not apply.
- **Vendor SDK**: cloned alongside at `../i2rt` (read-only reference; we import
  nothing from it at runtime except the URDF/MJCF and config YAML).

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose.py --clear   # 7/7 motors expected
```

---

## 2. What is verified on this hardware

| Fact | Evidence |
|---|---|
| All 7 motors online, feedback at CAN id + 0x10 | repeated `diagnose.py` runs |
| Position scale (±12.5 rad full scale) is correct | joint6 hit ±119.87° against the model's ±120° hard stop; joint4 sat at −97.3° against −97.0° |
| Hold-in-place works | errors ≤0.02° on unloaded joints |
| Gravity model of the *controller* is consistent | joint4 drooped 3.58° under 1.26 Nm; kp·error predicted 1.25 Nm |
| Jaw travel | 89.0 mm open → 4.9 mm closed, matching an independent hardware measurement |
| Control loop rate | 10.89 ms median per 6-joint tick = 91.8 Hz, resyncs 0, failures 0 |

## 3. Bugs found and fixed (do not reintroduce)

1. **`tip_position()` is not the jaw tips.** It returns the gripper *frame
   origin*, 134 mm behind the jaws. Every enrolled point was 134 mm out — four
   times the collision margin. Use `probe_position()`. Old sessions are
   recoverable via `recompute_positions()` because joint angles were stored.
2. **CAN frame desynchronisation.** The adapter echoes every transmission, so
   the RX queue is never empty. `_exchange` read without resynchronising and
   could accept a *stale* frame carrying the awaited id — silently returning a
   joint angle from a different moment. Now drains before every attempt.
3. **Blocking drain cost 5× the control rate.** `recv()` waits the full timeout
   on an empty queue. Pre-send drains are non-blocking now.
4. **`usb.reset()` breaks reopening on macOS.** It re-enumerates the adapter and
   invalidates the handle, so the first open worked and every reopen failed with
   "No such device". This masqueraded as a flaky adapter for hours.
5. **Motors latch error 0xD** when left enabled without a command stream, and
   disable/enable does NOT clear it. Needs `CLEAR_ERROR` (0xFB). `enable()`
   clears and retries per motor.
6. **Slew limiting was per tick**, so the real speed ceiling tracked whatever
   rate the loop achieved (114°/s at 100 Hz, 22°/s at 19 Hz). Now rad/s × elapsed.
7. **Sag verification tested impossible poses.** It perturbed joints 2 and 3
   negative, past their 0.0 hard stops. Clamped to joint limits now.
8. **`save()` ran before obstacles were added** in `build_map.py`, so the map on
   disk lacked the clamps and the table fill while the printed voxel count came
   from the in-memory copy.
9. **Canvas sizing.** `renderer.setSize(w, h, false)` skips CSS, and `inset:0`
   does not size a canvas, so both viewers rendered at device-pixel scale and
   displayed only the top-left quadrant.

## 4. Architecture

```
yam/can_compat.py       macOS gs_usb fixes, bus open with retries
yam/dm_motor.py         DM protocol encode/decode
yam/arm.py              YamArm: limits, torque ceiling, slew cap, error recovery
yam/kinematics.py       FK from URDF, probe point, collision spheres, IK
yam/voxel_map.py        occupancy + exact Euclidean distance field
yam/environment.py      ArmSafetyChecker: voxel env + MuJoCo self-collision
yam/planner.py          RRT-Connect + shortcutting + verify_under_tracking_error
yam/execution.py        GuardedExecutor: per-joint torque residual, tracking, temp
yam/lidar.py            PLY/OBJ/STL load, Kabsch, robot self-filtering
yam/scan_registration.py  arm-shape ICP registration (SEE SECTION 5 — SUSPECT)
yam/enrollment.py       touch capture, direction coverage
yam/viz_server.py       HTTP server for the phone app
ios/YamEnroll/          native SwiftUI app: ARKit LiDAR + enrollment
scripts/                diagnose, monitor, hold, enroll, build_map, plan_and_run,
                        pick_seed, preview_plan
```

Collision checking is deliberately two independent models: MuJoCo's exact convex
meshes for self-collision (the shipped MJCF has **no gripper geometry at all**),
and conservative spheres fitted to the URDF meshes for the environment. A pose is
free only if both agree.

---

## 5. THE BLOCKING PROBLEM: scan registration is wrong

The LiDAR scan is the right source of truth for the workcell — it is the actual
3D model. It is useless until it is registered into the robot frame, and **the
current registration is wrong.**

Decisive evidence, found last:

- Touched floor points (from the arm, sub-cm) sit at **x = +0.31…+0.50**, i.e.
  forward of the arm, off the table edge. This matches the real setup: the arm
  is mounted at the table edge facing off it.
- The registered scan puts its floor centred at **x = −1.34** and its tabletop at
  **x = −0.52**.
- These disagree by roughly **180° of yaw**.

Why the earlier "confirmation" was wrong: four seeds converged to the same base
position within 21 mm, but all four clicks were within ~20 cm of each other, so
they fall into the same basin of attraction. Agreement between non-independent
seeds is not evidence. `is_trustworthy` was also widened to 30 mm to accept a fit
scoring 19.6 mm, which lets in wrong yaws previously measured at 21–22 mm; it now
returns `verdict` ∈ {good, inconclusive, bad} and refuses to resolve that band.

**What to try next**, roughly in order of expected value:

1. **Constrain yaw with the touched points.** The 9 touched points are known to
   sub-cm in the robot frame and lie on real surfaces. Solve for the transform
   that puts them on scan surfaces *and* fits the arm shape, jointly. Yaw is the
   only badly-determined DoF (gravity fixes the rest), so this is a 1-D problem
   once the height is pinned.
2. **Seed from genuinely separated points** — click the base, then the far end of
   the arm, and require agreement between those.
3. **Sanity-gate any registration** on the touched-vs-scan disagreement above:
   the floor the arm touched and the floor in the scan must land on the same
   side. That check is cheap and would have caught this immediately.

## 6. Other open questions (raised by the operator, unresolved)

- **The base sits 11 mm below the tabletop.** The touched table height is
  z = +0.011 while the robot base origin is z = 0. The arm is bolted *on top of*
  the table, so these should coincide. 11 mm is the size of the FK + touch error,
  but the table slab is currently filled with its *top* at +0.011, which buries
  the base plate. Decide whether the mounting plane or the touched height is the
  datum, and fill accordingly.
- **The clamp model is wrong.** Given as 5.5 × 1 × 4 in, 102 mm either side of
  the base centreline, the arm's *home pose* reports as colliding with them
  (a link1 sphere penetrating 25.6 mm). Clamps holding that base cannot
  intersect the arm, so either the height (is 5.5 in vertical, or lying down?)
  or the position is wrong.
- **The scan has no tabletop within 0.47 m of the base** — the arm occludes it.
  Currently filled from the touched points. That fill is only as good as the
  registration and the datum question above.

## 7. Safety state — read before moving anything

- **Nothing has executed a planned path.** `plan_and_run.py` refuses to execute
  when the sag check fails, and it currently fails.
- Being inside the joint limits does **not** mean a pose is safe: ~10% of random
  in-limit poses self-collide (independently measured at 9.7% by a second
  session).
- **joint2 is the trap.** From the folded home pose it moves the tip *down*
  (0.165 → 0.012 m) and self-collides against the base past ~105°. joint3 is the
  joint that lifts.
- Gravity load is large: joint3 draws ~9.5 Nm lifting the arm's own weight, and
  12.36 Nm has been measured under load. A flat torque ceiling therefore cannot
  detect contact. The guard compares each joint against a slow baseline of its
  own torque and treats the *step* as contact; allowances are per-joint, scaled
  from a measured 7234-sample profile.
- **Do not build gravity feedforward on MuJoCo's `qfrc_bias`.** The shipped
  inertial model disagrees with this hardware in both directions (6.91 Nm
  modelled vs 0.02 Nm measured in one pose; 3.83 vs 7.66 in another). The
  collision geometry is trustworthy; the mass properties are not.
- The arm goes limp on every exit path, including Ctrl-C. Support it.
- Recorded resting pose, stable unpowered:
  `[0.0498, -0.0002, 0.0002, -0.0906, 0.0734, 1.1706]`

## 8. Still unvalidated on hardware

- `GuardedExecutor` has never fired on a real contact; the 2.5 Nm-class residual
  allowances are derived from a torque profile, not from a deliberate collision.
- FK absolute accuracy in metres. `scripts/enroll.py --validate-fk` touches one
  fixed point from several poses and reports the spread — that spread is the
  error budget every enrolled obstacle inherits. It has never been run.
- The iOS app's tap-to-align path end to end.
