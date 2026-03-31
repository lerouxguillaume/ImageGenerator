# Portrait Images

Place portrait images here. The game loads them at startup; missing files fall back to the built-in geometric art.

Portraits represent the **adventurer as a person**, independent of their class.

## Naming convention

`portrait_{index}.png` — e.g. `portrait_0.png`, `portrait_1.png`, ..., `portrait_11.png`

Each adventurer picks a random index at creation and keeps it permanently, even when reclassing.

## Pool size

Controlled by `PORTRAIT_POOL_SIZE` in `constants.hpp` (currently `12`).

## Recommended size

**120 × 128 px**. The game scales images to fill the portrait box (120 × 150 px), reserving the bottom 22 px for the class label bar.
