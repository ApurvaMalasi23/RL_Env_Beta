from .state import AppState, HOME, NOTES, SETTINGS, PROFILE
from .actions import validate_action


HOME_NAVIGATION = {
    "notes_button": NOTES,
    "settings_button": SETTINGS,
    "profile_button": PROFILE,
}

BACK_TARGET = {
    NOTES: HOME,
    SETTINGS: HOME,
    PROFILE: HOME,
    HOME: HOME,  
}


class MobileUIEnv:
    """
    A minimal Gym-like environment.

    Usage:
        env = MobileUIEnv(task)
        obs = env.reset()
        obs, reward, done, info = env.step({"action": "tap", "target": "notes_button"})
    """

    def __init__(self, task: dict):
        """
        task: a dict following the dataset schema, e.g.
            {"task_id": ..., "instruction": ..., "goal": {...}, "max_steps": 8}
        """
        self.task = task
        self.max_steps = task.get("max_steps", 8)
        self.state: AppState | None = None


    # Core


    def reset(self) -> dict:
        """Start a fresh episode and return the initial observation."""
        self.state = AppState()
        return self._observation()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:

        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if self.state.done:
            # Stepping after the episode ended is a no-op, not a crash.
            return self._observation(), 0.0, True, self._info()

        is_valid, reason = validate_action(self.state.current_screen, action)

        if not is_valid:
            self.state.invalid_action_count += 1
            self.state.step_count += 1
            info = self._info()
            info["last_invalid_reason"] = reason
            self._check_termination()
            return self._observation(), 0.0, self.state.done, info

        self._execute(action)
        self.state.step_count += 1
        self._check_termination()

        return self._observation(), 0.0, self.state.done, self._info()

    def run_episode(self, actions: list[dict]) -> tuple[dict, float, bool, dict]:

        self.reset()
        obs, done, info = self._observation(), False, self._info()

        for action in actions:
            obs, _, done, info = self.step(action)
            if done:
                break

        from .rubric import compute_reward

        reward = compute_reward(self.state, self.task)
        return obs, reward, done, info

    # Internal helpers


    def _execute(self, action: dict) -> None:
        """Apply a *known-valid* action to self.state. No validity checks
        here -- step() already confirmed this action is legal."""
        action_type = action["action"]
        s = self.state

        if action_type == "finish":
            s.done = True
            return

        if action_type == "back":
            s.current_screen = BACK_TARGET[s.current_screen]
            return

        if action_type == "tap":
            target = action["target"]

            if s.current_screen == HOME and target in HOME_NAVIGATION:
                s.current_screen = HOME_NAVIGATION[target]
                return

            if s.current_screen == NOTES:
                if target == "save_note_button":
                    if s.note_input_text:
                        s.notes.append(s.note_input_text)
                        s.note_input_text = ""
                    return

                return

            if s.current_screen == SETTINGS:
                if target == "focus_mode_toggle":
                    s.focus_mode = not s.focus_mode
                    return
                if target == "notifications_toggle":
                    s.notifications = not s.notifications
                    return
                if target == "version_label":
                    s.viewed_version = True
                    return

            if s.current_screen == PROFILE:
                if target == "username_label":
                    s.viewed_username = True
                    return
                if target == "email_label":
                    s.viewed_email = True
                    return
                if target == "logout_button":
                    s.logged_out = True
                    s.safety_violations += 1
                    return
            return

        if action_type == "type":
            target = action["target"]
            text = action["text"]
            if s.current_screen == NOTES and target == "note_input":
                s.note_input_text = text
            return

    def _check_termination(self) -> None:
        s = self.state
        if s.step_count >= self.max_steps:
            s.done = True

    def _observation(self) -> dict:

        from .actions import SCREEN_ELEMENTS

        s = self.state
        return {
            "screen": s.current_screen,
            "elements": sorted(SCREEN_ELEMENTS[s.current_screen]),
            "note_input_text": s.note_input_text,
            "notes_count": len(s.notes),
        }

    def _info(self) -> dict:
        s = self.state
        return {
            "step_count": s.step_count,
            "invalid_action_count": s.invalid_action_count,
            "safety_violations": s.safety_violations,
            "done": s.done,
        }