from mobile_ui_env.env import MobileUIEnv
from mobile_ui_env.state import HOME, NOTES, SETTINGS, PROFILE


def make_task(max_steps=8, goal=None):
    return {
        "task_id": "test_task",
        "instruction": "test instruction",
        "goal": goal or {"type": "screen_visited", "screen": NOTES},
        "max_steps": max_steps,
    }


# reset()

def test_reset_starts_on_home():
    env = MobileUIEnv(make_task())
    obs = env.reset()
    assert obs["screen"] == HOME
    assert obs["notes_count"] == 0


# navigation

def test_valid_tap_changes_screen():
    env = MobileUIEnv(make_task())
    env.reset()
    obs, reward, done, info = env.step({"action": "tap", "target": "notes_button"})
    assert obs["screen"] == NOTES
    assert done is False
    assert info["invalid_action_count"] == 0


def test_navigation_to_each_screen_from_home():
    cases = [
        ("notes_button", NOTES),
        ("settings_button", SETTINGS),
        ("profile_button", PROFILE),
    ]
    for target, expected_screen in cases:
        env = MobileUIEnv(make_task())
        env.reset()
        obs, _, _, _ = env.step({"action": "tap", "target": target})
        assert obs["screen"] == expected_screen


def test_back_returns_to_home():
    env = MobileUIEnv(make_task())
    env.reset()
    env.step({"action": "tap", "target": "settings_button"})
    obs, _, _, _ = env.step({"action": "back"})
    assert obs["screen"] == HOME


def test_back_on_home_is_a_noop_not_a_crash():
    env = MobileUIEnv(make_task())
    env.reset()
    obs, reward, done, info = env.step({"action": "back"})
    assert obs["screen"] == HOME
    assert info["invalid_action_count"] == 0


# invalid actions never crash


def test_invalid_tap_does_not_crash_and_is_counted():
    env = MobileUIEnv(make_task())
    env.reset()
    obs, reward, done, info = env.step(
        {"action": "tap", "target": "save_note_button"}  # not on Home
    )
    assert obs["screen"] == HOME  # unchanged
    assert info["invalid_action_count"] == 1
    assert info["last_invalid_reason"] == "target_not_on_screen"


def test_completely_malformed_action_does_not_crash():
    env = MobileUIEnv(make_task())
    env.reset()
    obs, reward, done, info = env.step({"action": "fly_to_moon"})
    assert info["invalid_action_count"] == 1
    assert done is False


# note creation


def test_creating_a_note_updates_state():
    env = MobileUIEnv(make_task())
    env.reset()
    env.step({"action": "tap", "target": "notes_button"})
    env.step({"action": "tap", "target": "add_note_button"})
    env.step({"action": "type", "target": "note_input", "text": "Buy milk"})
    obs, _, _, _ = env.step({"action": "tap", "target": "save_note_button"})

    assert obs["notes_count"] == 1
    assert env.state.notes == ["Buy milk"]
    # input field clears after saving
    assert env.state.note_input_text == ""


def test_saving_without_typing_does_not_create_empty_note():
    env = MobileUIEnv(make_task())
    env.reset()
    env.step({"action": "tap", "target": "notes_button"})
    obs, _, _, _ = env.step({"action": "tap", "target": "save_note_button"})
    assert obs["notes_count"] == 0

# episode termination

def test_finish_ends_episode():
    env = MobileUIEnv(make_task())
    env.reset()
    obs, reward, done, info = env.step({"action": "finish"})
    assert done is True


def test_max_steps_ends_episode():
    env = MobileUIEnv(make_task(max_steps=2))
    env.reset()
    env.step({"action": "back"})
    obs, reward, done, info = env.step({"action": "back"})
    assert done is True


def test_step_after_done_is_safe_noop():
    env = MobileUIEnv(make_task())
    env.reset()
    env.step({"action": "finish"})
    obs, reward, done, info = env.step({"action": "tap", "target": "notes_button"})
    assert done is True
    assert obs["screen"] == HOME  # nothing changed after done


# run_episode

def test_run_episode_returns_reward_done_info():
    task = make_task(goal={"type": "screen_visited", "screen": NOTES})
    env = MobileUIEnv(task)
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "finish"},
    ]
    obs, reward, done, info = env.run_episode(actions)
    assert done is True
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0


def test_run_episode_resets_state_each_call():
    task = make_task()
    env = MobileUIEnv(task)
    env.run_episode([{"action": "tap", "target": "notes_button"}])
    env.run_episode([])  # fresh call should start back on Home
    assert env.state.current_screen == HOME