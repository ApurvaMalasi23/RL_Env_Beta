# Mobile UI Agent Environment

A small RL-style environment simulating a mock mobile app (Home, Notes,
Settings, Profile screens) where an agent completes tasks via structured
`tap` / `type` / `back` / `finish` actions and receives a shaped reward.

## Setup

```bash
pip install -e .
pytest
python run_eval.py
```

No external services, databases, or API keys are required. Everything
runs locally with the standard library plus pytest.

---

## 1. What is the state space?

State is represented by the `AppState` dataclass in `state.py`:

- `current_screen` — one of `home`, `notes`, `settings`, `profile`
- `notes` — list of saved note strings
- `note_input_text` — text currently typed but not yet saved
- `focus_mode`, `notifications` — boolean toggles
- `username`, `email`, `app_version` — static ground-truth profile/settings values
- `viewed_username`, `viewed_email`, `viewed_version` — booleans tracking
  whether the agent has actually *looked at* each value (set when the
  corresponding label is tapped), separate from the value itself
- `logged_out` — whether the logout button has been tapped
- `step_count`, `invalid_action_count`, `safety_violations`, `done` —
  episode bookkeeping

The `viewed_*` flags are the one non-obvious part of the state space:
without them, an agent could be rewarded for "reporting" a username it
never actually read off the Profile screen, just by chance. They exist
specifically to make `username_viewed` / `email_viewed` / `version_viewed`
goals check *behavior* (did you navigate there and look), not just *luck*
(did the final state happen to match).

Episode counters (`step_count`, `invalid_action_count`,
`safety_violations`) live on `AppState` alongside the app's actual data,
which is a simplicity tradeoff — see Question 10.

## 2. What is the action space?

Four action types, all expressed as dicts (e.g. as if parsed from JSON):

```json
{"action": "tap", "target": "notes_button"}
{"action": "type", "target": "note_input", "text": "Buy milk"}
{"action": "back"}
{"action": "finish"}
```

`tap` is valid only if `target` is an element that exists on the
*current* screen (e.g. `save_note_button` only exists on Notes).
`type` is further restricted to `note_input` specifically — it's the only
text field in this app, so typing into a button or label is invalid even
though the target itself exists on the screen.
`back` always returns to Home (or is a no-op on Home).
`finish` always ends the episode immediately.

Invalid actions (wrong screen, unknown target, malformed dict, unknown
action type) are never allowed to crash the environment — they're counted
in `invalid_action_count` and the episode continues normally.

## 3. What is the episode termination condition?

An episode ends when either:

- the agent calls `{"action": "finish"}`, or
- `step_count` reaches the task's `max_steps`

Both push `state.done = True`. Calling `step()` again after `done` is a
safe no-op (returns the same observation, `done=True`, no further
state change) rather than raising.

## 4. Which rewards are sparse?

**`success_reward`** is the sparse, terminal signal: 1.0 if the task's
goal is satisfied by the final state, 0.0 otherwise. It only resolves at
the end of the episode and gives no signal about *how close* the agent
got while acting — this is the textbook "sparse reward" problem: an agent
that navigates correctly through 3 of 4 required steps and one that does
nothing at all both score 0 on this term alone.

## 5. Which rewards are dense or shaped?

The other five components are shaped/dense, computed from intermediate
behavior rather than only the final goal check:

- **`format_reward`** — fraction of actions taken that were valid (not
  just well-formed JSON, since by the time `step()` sees an action it's
  already a parsed dict in this implementation — see Question 10)
- **`efficiency_reward`** — rewards finishing in fewer of the allotted
  steps, but only applied when the task succeeded (see Question 6 for why)
- **`partial_progress_reward`** — gives credit for being on the screen
  relevant to the goal at episode end, even if the full goal wasn't
  completed — this directly targets the sparse-reward problem from
  Question 4, so an agent that reaches Settings but runs out of steps
  before tapping a toggle gets *some* signal, not the same zero as an
  agent that never left Home
- **`invalid_action_penalty`** / **`safety_penalty`** — per-event
  penalties (counts), applied continuously rather than only at the end

## 6. How can reward hacking happen in this environment?

A few concrete ways, found while building and testing this environment
(not theoretical):

- **Reward ceiling saturation.** The formula's positive terms
  (`success + 0.1*format + 0.2*efficiency + 0.1*partial_progress`) can sum
  above 1.0 before penalties (e.g. `1.0 + 0.1 + 0.2 + 0.1 = 1.4`). After
  `clamp(0, 1)`, a single `-0.1` invalid-action penalty on an otherwise
  clean run is fully absorbed by that headroom — the final reward is
  identical to a perfectly clean run. This means an agent that's "mostly
  careful" pays no visible price for one sloppy action, which an RL
  algorithm could learn to exploit (sloppy exploration becomes free as
  long as the task still succeeds). Documented and tested explicitly in
  `test_rewards.py::test_compute_reward_ceiling_saturation_with_single_invalid_action`.
- **`efficiency_reward` gating.** If efficiency were rewarded regardless
  of success, an agent could learn to call `finish` immediately every
  episode to harvest free efficiency reward without ever attempting the
  task. `compute_reward` only applies efficiency when `success == 1.0` to
  block this.
- **`partial_progress_reward`'s "ended there" semantics.** It checks the
  screen at episode *end*, not whether the screen was *ever* visited. An
  agent that correctly reaches the target screen, then wanders off before
  the episode ends, gets no partial credit despite having done the right
  thing at some point. This is a real, documented limitation — not
  patched, because the alternative ("ever visited") is also exploitable
  (an agent could flicker through every screen once just to bank partial
  credit on every goal type, regardless of relevance to the actual task).
- **`viewed_*` flags vs. raw values.** This is reward-hacking *prevention*
  rather than a hole: without these flags, an agent could trivially "win"
  `username_viewed` goals by guessing/hardcoding the right string without
  ever navigating to Profile.

## 7. How would you scale this from a mock UI to a real Android emulator?

The state/action/reward separation is designed to make this swap
contained rather than a rewrite:

- `AppState` would be replaced by a real **accessibility tree** snapshot
  pulled from the emulator (e.g. via `adb` / UIAutomator / Appium), plus
  optionally a screenshot — the dataclass fields here are a hand-rolled
  stand-in for that tree.
- `actions.py`'s `validate_action` + `SCREEN_ELEMENTS` lookup would be
  replaced by querying the live accessibility tree for which elements
  are actually present and interactable, instead of a hardcoded dict.
- `env.py`'s `_execute` would be replaced by an **action executor** that
  sends real taps/text-input/back/Home-button events to the device via
  `adb shell input` or an Appium driver, instead of directly mutating a
  Python object.
- `rubric.py`'s goal checkers would need to read the *real* app's UI
  state or its underlying database/SharedPreferences to verify outcomes,
  instead of reading dataclass fields directly.
- The dataset and reward formula structure (goal types, weighted reward
  components, clamping) would carry over largely unchanged — that part of
  the design doesn't depend on the UI being mocked.

## 8. How would this work with Prime Intellect, Verifiers, or PRIME-RL later?

This project includes a deliberately minimal local `Rubric` class
(`rubric.py`) standing in for `verifiers.Rubric` — same shape (named
reward functions + weights), without the full framework. A real
integration would:

- Replace the local `Rubric` with `vf.Rubric(funcs=[...], weights=[...])`
  from the actual `verifiers` package.
- Wrap `MobileUIEnv` (or its Android-emulator successor) as a
  `vf.SingleTurnEnv` (or a multi-turn environment if the task requires
  back-and-forth interaction — see Question 9), exposing `dataset` and
  `eval_dataset` via `build_dataset()` exactly as already structured here.
- `load_environment()` is already written to the shape the assignment
  specifies, so swapping in real `verifiers` imports there is the main
  integration point — the dataset builder, goal checkers, and reward
  functions underneath wouldn't need to change.
- For PRIME-RL specifically, this environment's reward signal
  (`compute_reward`, bounded to `[0, 1]`) is the kind of scalar reward a
  PRIME-RL training loop would consume per rollout; the main additional
  work would be wiring up a real policy (vs. the rule-based baseline
  here) and batching multiple environment instances for parallel rollout
  collection.

## 9. What tests did you write?

60 pytest tests across three files:

- **`tests/test_actions.py`** — `validate_action` in isolation: valid taps
  on every screen, invalid taps (wrong screen, unknown target, missing
  target), valid/invalid `type` actions (including the `note_input`-only
  restriction), `back`/`finish` always valid, malformed/unknown action
  types, and the small helper functions directly.
- **`tests/test_env.py`** — `MobileUIEnv` behavior: reset starting state,
  navigation from Home to each screen, `back` behavior (including the
  Home-is-a-no-op edge case), invalid actions never crashing and being
  counted, note creation (including the edge case of saving with empty
  input not creating a blank note), both termination paths (`finish` and
  `max_steps`), stepping after `done` being a safe no-op, and
  `run_episode`'s return shape.
- **`tests/test_rewards.py`** — goal verification for all 9 goal types
  (including an explicit test that `username_viewed` requires the
  `viewed_username` flag, not just a matching value) and each of the six
  reward functions tested independently against a hand-built `AppState`,
  plus integration tests through full episodes: clean success, logout
  triggering the safety penalty, invalid actions reducing reward, reward
  never going below 0 or above 1, and the ceiling-saturation behavior
  from Question 6 documented as its own explicit test rather than left
  as a surprise.

## 10. What tradeoffs did you make due to the limited assignment scope?

- **Episode counters live on `AppState`** rather than a separate
  `EpisodeState` object composed alongside it. Simpler to pass around as
  one object, but blurs "the app's data" with "bookkeeping about this
  episode" — a cleaner separation would split these.
- **`format_reward` measures "fraction of valid actions taken," not
  literal JSON well-formedness.** Actions arrive as already-parsed dicts
  in this implementation, so there's no raw-JSON-parsing step to validate
  separately from semantic validity (right target, right screen). A more
  faithful version would validate raw JSON text before dict conversion
  and penalize syntax errors separately from valid-JSON-but-wrong-target
  actions.
- **`partial_progress_reward` checks the final screen, not visit
  history**, as discussed in Question 6 — a deliberate simplicity choice,
  not an oversight, but one that trades some shaped-reward generosity for
  resistance to a specific kind of gaming.
- **The local `Rubric` class is a thin facade**, not a fully generic
  scorer. `compute_reward` (not `Rubric.score()`) is what actually runs,
  because the six reward functions take different argument shapes (some
  need `goal`, some need `max_steps`, some need raw counts) and a fully
  generic "call every function with the same arguments" loop would need
  extra plumbing this assignment's scope didn't require.
- **The rule-based baseline agent in `run_eval.py` is intentionally
  simple keyword matching**, not real language understanding. It passes
  100% of eval tasks as written, but stress-testing it against rephrased
  instructions (documented in `AI_USAGE.md`) shows it genuinely fails on
  phrasing it wasn't designed around (e.g. "jot down a reminder" instead
  of "create a note"). `run_eval.py` prints this caveat explicitly so the
  100% figure isn't mistaken for demonstrated generalization.
- **Screens are string constants, not an Enum.** Slightly less type-safe,
  but simpler to serialize, print, and reason about for a project this
  size — and consistent with the "no overengineering" goal.
