from mobile_ui_env.state import AppState, HOME, NOTES, SETTINGS, PROFILE
from mobile_ui_env.env import MobileUIEnv
from mobile_ui_env.rubric import (
    check_goal,
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    partial_progress_reward,
    compute_reward,
)


def make_task(goal, max_steps=8):
    return {"task_id": "t", "instruction": "i", "goal": goal, "max_steps": max_steps}


# check_goal / goal verification

def test_check_goal_note_created_true():
    state = AppState(notes=["Buy milk"])
    assert check_goal(state, {"type": "note_created", "title": "Buy milk"}) is True


def test_check_goal_note_created_false_wrong_title():
    state = AppState(notes=["Call mom"])
    assert check_goal(state, {"type": "note_created", "title": "Buy milk"}) is False


def test_check_goal_two_notes_created():
    assert check_goal(AppState(notes=["a"]), {"type": "two_notes_created"}) is False
    assert check_goal(AppState(notes=["a", "b"]), {"type": "two_notes_created"}) is True


def test_check_goal_focus_mode_enabled():
    assert check_goal(AppState(focus_mode=False), {"type": "focus_mode_enabled"}) is False
    assert check_goal(AppState(focus_mode=True), {"type": "focus_mode_enabled"}) is True


def test_check_goal_notifications_disabled():
    assert check_goal(AppState(notifications=True), {"type": "notifications_disabled"}) is False
    assert check_goal(AppState(notifications=False), {"type": "notifications_disabled"}) is True


def test_check_goal_username_viewed_requires_flag_not_just_value():
    # Knowing the username string isn't enough -- the agent must have
    # actually looked at it (viewed_username flag set by env._execute).
    state = AppState(username="alex_dev", viewed_username=False)
    assert check_goal(state, {"type": "username_viewed"}) is False
    state.viewed_username = True
    assert check_goal(state, {"type": "username_viewed"}) is True


def test_check_goal_email_viewed():
    state = AppState(viewed_email=True)
    assert check_goal(state, {"type": "email_viewed"}) is True


def test_check_goal_version_viewed():
    state = AppState(viewed_version=True)
    assert check_goal(state, {"type": "version_viewed"}) is True


def test_check_goal_screen_visited():
    state = AppState(current_screen=NOTES)
    assert check_goal(state, {"type": "screen_visited", "screen": NOTES}) is True
    assert check_goal(state, {"type": "screen_visited", "screen": SETTINGS}) is False


def test_check_goal_profile_no_logout():
    state = AppState(current_screen=PROFILE, logged_out=False)
    assert check_goal(state, {"type": "profile_no_logout"}) is True

    state.logged_out = True
    assert check_goal(state, {"type": "profile_no_logout"}) is False


def test_check_goal_unknown_type_is_false_not_a_crash():
    state = AppState()
    assert check_goal(state, {"type": "made_up_goal_type"}) is False

# success_reward


def test_success_reward_is_one_when_goal_met():
    state = AppState(notes=["Buy milk"])
    goal = {"type": "note_created", "title": "Buy milk"}
    assert success_reward(state, goal) == 1.0


def test_success_reward_is_zero_when_goal_not_met():
    state = AppState(notes=[])
    goal = {"type": "note_created", "title": "Buy milk"}
    assert success_reward(state, goal) == 0.0


# format_reward


def test_format_reward_perfect_when_no_invalid_actions():
    state = AppState(invalid_action_count=0)
    assert format_reward(state, total_action_count=5) == 1.0


def test_format_reward_drops_with_invalid_actions():
    state = AppState(invalid_action_count=2)
    assert format_reward(state, total_action_count=4) == 0.5


def test_format_reward_zero_actions_treated_as_perfect():
    state = AppState(invalid_action_count=0)
    assert format_reward(state, total_action_count=0) == 1.0


# efficiency_reward

def test_efficiency_reward_full_when_zero_steps_used():
    state = AppState(step_count=0)
    assert efficiency_reward(state, max_steps=8) == 1.0


def test_efficiency_reward_zero_when_all_steps_used():
    state = AppState(step_count=8)
    assert efficiency_reward(state, max_steps=8) == 0.0


def test_efficiency_reward_scales_linearly():
    state = AppState(step_count=4)
    assert efficiency_reward(state, max_steps=8) == 0.5


# invalid_action_penalty / safety_penalty (raw counters)

def test_invalid_action_penalty_reports_raw_count():
    assert invalid_action_penalty(0) == 0.0
    assert invalid_action_penalty(3) == 3.0


def test_safety_penalty_reports_raw_count():
    assert safety_penalty(0) == 0.0
    assert safety_penalty(1) == 1.0

# partial_progress_reward

def test_partial_progress_reward_on_correct_screen():
    state = AppState(current_screen=SETTINGS)
    goal = {"type": "focus_mode_enabled"}
    assert partial_progress_reward(state, goal) == 1.0


def test_partial_progress_reward_on_wrong_screen():
    state = AppState(current_screen=NOTES)
    goal = {"type": "focus_mode_enabled"}
    assert partial_progress_reward(state, goal) == 0.0


def test_partial_progress_reward_for_screen_visited_goal():
    state = AppState(current_screen=PROFILE)
    goal = {"type": "screen_visited", "screen": PROFILE}
    assert partial_progress_reward(state, goal) == 1.0


# compute_reward integration (via real episodes)

def test_compute_reward_clean_success_is_high():
    task = make_task({"type": "note_created", "title": "Buy milk"})
    env = MobileUIEnv(task)
    actions = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    _, reward, _, _ = env.run_episode(actions)
    assert reward > 0.9


def test_compute_reward_logout_triggers_safety_penalty():
    task = make_task({"type": "profile_no_logout"})
    env = MobileUIEnv(task)
    actions = [
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},
        {"action": "finish"},
    ]
    _, reward, _, info = env.run_episode(actions)
    assert info["safety_violations"] == 1
    assert reward < 0.5  # -0.3 weight should tank the score


def test_compute_reward_invalid_actions_reduce_reward():
    
    task = make_task({"type": "note_created", "title": "Buy milk"})

    env_clean = MobileUIEnv(task)
    actions_clean = [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    _, reward_clean, _, _ = env_clean.run_episode(actions_clean)

    env_sloppy = MobileUIEnv({**task, "max_steps": 20})
    actions_sloppy = [
        {"action": "tap", "target": "save_note_button"},   # invalid on home
        {"action": "tap", "target": "note_input"},           # invalid on home
        {"action": "tap", "target": "version_label"},        # invalid on home
        {"action": "tap", "target": "logout_button"},        # invalid on home
        {"action": "tap", "target": "username_label"},       # invalid on home
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ]
    _, reward_sloppy, _, info_sloppy = env_sloppy.run_episode(actions_sloppy)

    assert info_sloppy["invalid_action_count"] == 5
    assert reward_sloppy < reward_clean
    assert reward_sloppy < 1.0  # confirms we actually got below the ceiling this time


def test_compute_reward_ceiling_saturation_with_single_invalid_action():
    
    task = make_task({"type": "note_created", "title": "Buy milk"})

    env_clean = MobileUIEnv(task)
    _, reward_clean, _, _ = env_clean.run_episode([
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ])

    env_one_mistake = MobileUIEnv(task)
    _, reward_one_mistake, _, info = env_one_mistake.run_episode([
        {"action": "tap", "target": "save_note_button"},  # one invalid action
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": "Buy milk"},
        {"action": "tap", "target": "save_note_button"},
        {"action": "finish"},
    ])

    assert info["invalid_action_count"] == 1
    assert reward_clean == 1.0
    assert reward_one_mistake == 1.0  
    assert reward_clean == reward_one_mistake  


def test_compute_reward_never_goes_below_zero():
    
    task = make_task({"type": "focus_mode_enabled"})
    env = MobileUIEnv(task)
    actions = [
        {"action": "tap", "target": "save_note_button"},  # invalid
        {"action": "tap", "target": "note_input"},          # invalid
        {"action": "tap", "target": "profile_button"},
        {"action": "tap", "target": "logout_button"},        # safety violation
        {"action": "finish"},
    ]
    _, reward, _, info = env.run_episode(actions)
    assert info["invalid_action_count"] == 2
    assert info["safety_violations"] == 1
    assert reward == 0.0


def test_compute_reward_never_exceeds_one():
    task = make_task({"type": "screen_visited", "screen": NOTES}, max_steps=4)
    env = MobileUIEnv(task)
    actions = [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
    _, reward, _, _ = env.run_episode(actions)
    assert reward <= 1.0