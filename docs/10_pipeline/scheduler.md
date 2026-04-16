# Scheduler (DPM++ 2M Karras)

The scheduler controls timestep sampling during diffusion.

---

# Core function

- `buildKarrasSchedule()`

Produces:
- sigma sequence of length N+1
- last sigma = 0

---

# Sigma behavior

Each step:
- sigma decreases non-linearly
- Karras weighting improves perceptual quality

---

# sigma → timestep conversion

- `sigmaToTimestep()` maps sigma values into model timestep space

---

# Design purpose

The scheduler defines:
- noise reduction curve
- sampling stability
- image sharpness characteristics