from mobile_ui_env.actions import (
    validate_action,
    is_known_action_type,
    is_valid_target,
)
from mobile_ui_env.state import HOME, NOTES, SETTINGS, PROFILE


# tap


def test_valid_tap_on_home():
    ok, reason = validate_action(HOME, {"action": "tap", "target": "notes_button"})
    assert ok is True
    assert reason == "ok"


def test_valid_tap_on_each_screen():
    # Sanity check: every screen has at least one tappable element that
    # validates successfully.
    cases = [
        (HOME, "settings_button"),
        (NOTES, "add_note_button"),
        (SETTINGS, "focus_mode_toggle"),
        (PROFILE, "logout_button"),
    ]
    for screen, target in cases:
        ok, _ = validate_action(screen, {"action": "tap", "target": target})
        assert ok is True


def test_invalid_tap_wrong_screen():
    # "save_note_button" only exists on the Notes screen, not Home.
    ok, reason = validate_action(HOME, {"action": "tap", "target": "save_note_button"})
    assert ok is False
    assert reason == "target_not_on_screen"


def test_invalid_tap_unknown_target():
    ok, reason = validate_action(HOME, {"action": "tap", "target": "does_not_exist"})
    assert ok is False
    assert reason == "target_not_on_screen"


def test_invalid_tap_missing_target():
    ok, reason = validate_action(HOME, {"action": "tap"})
    assert ok is False
    assert reason == "missing_target"

# type


def test_valid_type_into_note_input():
    ok, reason = validate_action(
        NOTES, {"action": "type", "target": "note_input", "text": "Buy milk"}
    )
    assert ok is True
    assert reason == "ok"


def test_invalid_type_into_non_typeable_target():
    ok, reason = validate_action(
        NOTES, {"action": "type", "target": "note_list", "text": "hello"}
    )
    assert ok is False
    assert reason == "target_not_typeable"


def test_invalid_type_missing_text():
    ok, reason = validate_action(NOTES, {"action": "type", "target": "note_input"})
    assert ok is False
    assert reason == "missing_text"


def test_invalid_type_wrong_screen():
    ok, reason = validate_action(
        HOME, {"action": "type", "target": "note_input", "text": "hi"}
    )
    assert ok is False
    assert reason == "target_not_on_screen"

# back / finish


def test_back_always_valid():
    for screen in (HOME, NOTES, SETTINGS, PROFILE):
        ok, reason = validate_action(screen, {"action": "back"})
        assert ok is True
        assert reason == "ok"


def test_finish_always_valid():
    ok, reason = validate_action(HOME, {"action": "finish"})
    assert ok is True
    assert reason == "ok"


# unknown actions


def test_unknown_action_type():
    ok, reason = validate_action(HOME, {"action": "swipe", "target": "notes_button"})
    assert ok is False
    assert reason == "unknown_action_type"


def test_action_not_a_dict():
    ok, reason = validate_action(HOME, "tap notes_button")
    assert ok is False
    assert reason == "action_not_a_dict"


def test_action_missing_action_key():
    ok, reason = validate_action(HOME, {"target": "notes_button"})
    assert ok is False
    assert reason == "unknown_action_type"


# helper functions


def test_is_known_action_type():
    assert is_known_action_type({"action": "tap"}) is True
    assert is_known_action_type({"action": "swipe"}) is False
    assert is_known_action_type("not a dict") is False


def test_is_valid_target():
    assert is_valid_target(HOME, "notes_button") is True
    assert is_valid_target(HOME, "save_note_button") is False
    assert is_valid_target("nonexistent_screen", "anything") is False