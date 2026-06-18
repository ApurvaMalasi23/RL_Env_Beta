# AI Usage

This project was built with AI assistance (Claude) in an incremental,
file-by-file session rather than a single generated dump. This document
describes what was asked, what was accepted, what was changed, and what
I'd still want to review further.

## What I asked the AI

I asked Claude to help implement the Mobile UI Agent RL environment
assignment one file at a time, in dependency order: state -> actions ->
env -> tests for those -> dataset -> rubric -> tests for that -> eval
script. For each file, I asked for the design reasoning before or alongside
the code, specifically so I could understand and defend the choices rather
than just receive output.

I did not ask for the README or the overall task breakdown to be decided
for me; I drove the order of work and approved each step before moving on.

## What I accepted from the AI

- The overall module structure (`state.py`, `actions.py`, `env.py`,
  `dataset.py`, `rubric.py`) and the pattern of keeping validation
  (`actions.py`), execution (`env.py`), and reward computation
  (`rubric.py`) as separate concerns.
- The `AppState` dataclass design, including the `viewed_username` /
  `viewed_email` / `viewed_version` flags as a deliberate way to prevent
  an agent from getting credit for "reporting" a value it never actually
  looked at.
- The goal-verifier dispatch pattern in `rubric.py` (`goal["type"] ->
  checker function`), which keeps goal-checking generic instead of
  hardcoding per-task logic.
- The six reward component functions and the `compute_reward` combination
  formula, as specified by the assignment.
- The rule-based baseline agent in `run_eval.py` (keyword matching on
  instruction text -> action sequence).
- Generated pytest test files (`test_actions.py`, `test_env.py`,
  `test_rewards.py`), which I then ran and used to find real bugs (see
  below) rather than accepting on faith.

## What I changed / caught myself (with AI assistance debugging)

This was not a "generate once and ship" process. Several real issues
surfaced while building and testing, and were fixed rather than ignored:

1. **Dataset off-by-one.** The first version of `dataset.py` produced 11
   eval tasks instead of the required 10 (one extra note-title variant).
   Caught by writing a small schema/count sanity check and run immediately
   after generating the file, not discovered later.

2. **`partial_progress_reward` test bug vs. real behavior.** An early test
   used a `back` action to "burn a step" after reaching the right screen,
   which moved the agent back to Home before reward was computed --
   making it look like partial-progress reward wasn't distinguishing a
   near-miss from total failure. Investigating this surfaced a genuine,
   documented design limitation: `partial_progress_reward` checks the
   *final* screen at episode end, not "was this screen ever visited."
   That's now called out explicitly as a known scope tradeoff (see
   `rubric.py` docstring and the design-tradeoffs discussion below).

3. **Reward ceiling saturation.** A test asserting that one invalid action
   should reduce reward below a clean run's reward *failed*. Root cause:
   the assignment's reward formula's positive terms (`success + 0.1*format
   + 0.2*efficiency + 0.1*partial_progress`) can sum above 1.0 before
   penalties (e.g. 1.0 + 0.1 + 0.2 + 0.1 = 1.4), so the final `clamp(0, 1)`
   absorbs small penalties whenever the agent is otherwise doing well. This
   is a property of the given weights, not a bug in the implementation. It
   is now documented explicitly with a dedicated test
   (`test_compute_reward_ceiling_saturation_with_single_invalid_action`)
   that asserts and names this behavior instead of hiding it, plus the
   original test was fixed to use enough invalid actions to actually
   demonstrate a measurable reward reduction.

4. **Baseline agent "100% success" sanity check.** After `run_eval.py`
   reported 100% success on all 10 eval tasks, I treated that as a
   suspicious signal rather than a win, since the keyword rules and the
   dataset's instruction phrasing were written in the same sitting. I
   asked for (and got) a stress test using rephrased instructions the
   rules weren't written against (e.g. "jot down a reminder" instead of
   "create a note"). This confirmed the rules generalize on some
   rephrasings (username/email/focus-mode/notifications keyword matches
   still work) but genuinely fail on others (no "note" or "create"/
   "titled" keyword present), correctly falling back to an immediate
   `finish` rather than guessing. The eval script now prints an explicit
   caveat explaining that 100% baseline success reflects keyword/dataset
   alignment, not demonstrated generalization, so the output doesn't
   imply more capability than the baseline actually has.

## What I'd still want to review or modify further

- **`partial_progress_reward`'s "ended there" vs. "ever visited" choice.**
  Currently strict (final screen only). I'd want to decide deliberately
  whether to add a `visited_screens` set to `AppState` for more forgiving
  partial credit, weighing that against making the reward easier to game.
- **Reward formula's ceiling saturation**, documented above. If this were
  going into real training rather than an assignment, I'd want to either
  widen the clamp range, normalize the weights so they don't exceed 1.0 on
  the positive side, or keep the formula as specified, but with that
  property called out as accepted, not assumed away.
- **`format_reward`'s interpretation.** The assignment describes format
  reward in terms of "valid JSON," but in this environment actions arrive
  as already-parsed dicts, so `format_reward` measures "fraction of
  actions that were valid on their screen" rather than JSON well-formedness
  specifically. This is a reasonable simplification for this environment
  but worth being explicit about rather than implying it checks something
  it doesn't.
- **The keyword-matching baseline agent** is intentionally simple and not
  meant to be a strong baseline -- I would not want this mistaken for a
  trained or LLM-driven policy. It exists to prove the pipeline (dataset ->
  env -> reward) works end-to-end.

## What I learned

The most useful part of this process was discovering that test failures
were sometimes pointing at real properties of the reward design (ceiling
saturation, "ended there" vs. "ever visited" semantics) rather than simple
coding mistakes -- and that the right response was to document and test
for that behavior explicitly, not just patch the test until it passed.
That distinction (bug vs. designed-but-imperfect behavior) is something
I'd want to be able to explain clearly in review.
