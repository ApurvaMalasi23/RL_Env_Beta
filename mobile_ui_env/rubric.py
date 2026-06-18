from .state import AppState

# Goal verification


def _check_note_created(state: AppState, goal: dict) -> bool:
    title = goal.get("title")
    return title in state.notes


def _check_two_notes_created(state: AppState, goal: dict) -> bool:
    return len(state.notes) >= 2


def _check_focus_mode_enabled(state: AppState, goal: dict) -> bool:
    return state.focus_mode is True


def _check_notifications_disabled(state: AppState, goal: dict) -> bool:
    return state.notifications is False


def _check_username_viewed(state: AppState, goal: dict) -> bool:
    return state.viewed_username is True


def _check_email_viewed(state: AppState, goal: dict) -> bool:
    return state.viewed_email is True


def _check_version_viewed(state: AppState, goal: dict) -> bool:
    return state.viewed_version is True


def _check_screen_visited(state: AppState, goal: dict) -> bool:
   
    return state.current_screen == goal.get("screen")


def _check_profile_no_logout(state: AppState, goal: dict) -> bool:
    return state.current_screen == "profile" and state.logged_out is False



_GOAL_CHECKERS = {
    "note_created": _check_note_created,
    "two_notes_created": _check_two_notes_created,
    "focus_mode_enabled": _check_focus_mode_enabled,
    "notifications_disabled": _check_notifications_disabled,
    "username_viewed": _check_username_viewed,
    "email_viewed": _check_email_viewed,
    "version_viewed": _check_version_viewed,
    "screen_visited": _check_screen_visited,
    "profile_no_logout": _check_profile_no_logout,
}


def check_goal(state: AppState, goal: dict) -> bool:
    
    checker = _GOAL_CHECKERS.get(goal.get("type"))
    if checker is None:
        return False
    return checker(state, goal)


# Reward components

def success_reward(state: AppState, goal: dict) -> float:
    """1.0 if the task's goal is satisfied by the final state, else 0.0.
    This is the sparse, terminal, ground-truth signal -- everything else
    in the formula exists to make learning easier or to penalize bad
    behavior, not to define what success means."""
    return 1.0 if check_goal(state, goal) else 0.0


def format_reward(state: AppState, total_action_count: int) -> float:
    """
    Rewards the agent for producing well-formed actions overall. Defined
    as the fraction of actions that were valid (i.e. 1 - invalid rate).

    If zero actions were taken, we treat format as perfect (1.0) rather
    than dividing by zero -- there's no malformed output to penalize.
    """
    if total_action_count == 0:
        return 1.0
    valid_count = total_action_count - state.invalid_action_count
    return max(0.0, valid_count / total_action_count)


def efficiency_reward(state: AppState, max_steps: int) -> float:
    """
    Rewards completing the episode in fewer steps. Only meaningful when
    the task actually succeeded -- an agent that fails quickly shouldn't
    be rewarded for "efficiency". compute_reward() is responsible for
    only applying this when success_reward is 1.0; this function itself
    just computes the step-based ratio so it stays simple and testable.

    Linear scale: using 0 steps would be a perfect 1.0, using all
    max_steps gives 0.0.
    """
    if max_steps <= 0:
        return 0.0
    remaining = max_steps - state.step_count
    return max(0.0, min(1.0, remaining / max_steps))


def invalid_action_penalty(invalid_action_count: int) -> float:
    """Raw count of invalid actions taken. compute_reward applies the
    -0.1 weight; this function just reports the count so it's reusable
    and easy to assert against in tests."""
    return float(invalid_action_count)


def safety_penalty(safety_violations: int) -> float:
    """Raw count of safety violations (currently: logging out). Same
    pattern as invalid_action_penalty -- report the count, let
    compute_reward apply the weight."""
    return float(safety_violations)


def partial_progress_reward(state: AppState, goal: dict) -> float:
    """
    Optional shaped reward for making progress toward the goal even if
    the full goal wasn't completed -- this directly addresses "why is
    sparse reward hard for RL agents": without this, an agent that
    correctly navigates to Settings but runs out of steps before tapping
    the toggle gets *zero* signal, identical to an agent that wandered
    aimlessly. partial_progress distinguishes those two cases.

    Implemented as "did the agent reach the screen relevant to this
    goal", independent of whether the goal itself completed. Deliberately
    simple: a more elaborate version could give partial credit per
    subgoal (e.g. "typed something" vs "saved it"), but screen-reached is
    enough to demonstrate the concept without overengineering.
    """
    goal_type = goal.get("type")

    screen_for_goal = {
        "note_created": "notes",
        "two_notes_created": "notes",
        "focus_mode_enabled": "settings",
        "notifications_disabled": "settings",
        "version_viewed": "settings",
        "username_viewed": "profile",
        "email_viewed": "profile",
        "profile_no_logout": "profile",
    }.get(goal_type)

    if goal_type == "screen_visited":
        screen_for_goal = goal.get("screen")

    if screen_for_goal is None:
        return 0.0

    return 1.0 if state.current_screen == screen_for_goal else 0.0


# Combining everything

def compute_reward(state: AppState, task: dict) -> float:
    """
    Combine all reward components into the final scalar reward using the
    assignment-specified weighted formula, then clamp to [0, 1].

        reward = success
               + 0.1 * format
               + 0.2 * efficiency
               + 0.1 * partial_progress
               - 0.1 * invalid_actions
               - 0.3 * safety_violations

    Efficiency is only counted when the task succeeded -- rewarding "fast
    failure" would actively encourage giving up early, which is the
    opposite of what we want.
    """
    goal = task["goal"]
    max_steps = task.get("max_steps", 8)

    success = success_reward(state, goal)
    fmt = format_reward(state, state.step_count)
    efficiency = efficiency_reward(state, max_steps) if success == 1.0 else 0.0
    partial = partial_progress_reward(state, goal)
    invalid_count = invalid_action_penalty(state.invalid_action_count)
    safety_count = safety_penalty(state.safety_violations)

    raw = (
        success
        + 0.1 * fmt
        + 0.2 * efficiency
        + 0.1 * partial
        - 0.1 * invalid_count
        - 0.3 * safety_count
    )

    return max(0.0, min(1.0, raw))


# Minimal Verifiers-style Rubric

class Rubric:
   
    def __init__(self, funcs: list, weights: list[float]):
        if len(funcs) != len(weights):
            raise ValueError("funcs and weights must be the same length")
        self.funcs = funcs
        self.weights = weights

    def describe(self) -> list[tuple[str, float]]:
        """Returns [(function_name, weight), ...] -- useful for printing
        the rubric's composition in run_eval.py or debugging."""
        return [(f.__name__, w) for f, w in zip(self.funcs, self.weights)]