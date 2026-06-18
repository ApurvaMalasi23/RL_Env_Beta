from .state import HOME, NOTES, SETTINGS, PROFILE


SCREEN_ELEMENTS: dict[str, set[str]] = {
    HOME: {"notes_button", "settings_button", "profile_button"},
    NOTES: {"add_note_button", "note_input", "save_note_button", "note_list"},
    SETTINGS: {"focus_mode_toggle", "notifications_toggle", "version_label"},
    PROFILE: {"username_label", "email_label", "logout_button"},
}

# The only action "verbs" the environment understands.
VALID_ACTION_TYPES = {"tap", "type", "back", "finish"}


def is_known_action_type(action: dict) -> bool:
    """Does this action use one of the four supported verbs?"""
    return isinstance(action, dict) and action.get("action") in VALID_ACTION_TYPES


def is_valid_target(screen: str, target: str) -> bool:
    """Does `target` exist on `screen`?"""
    return target in SCREEN_ELEMENTS.get(screen, set())


def validate_action(screen: str, action: dict) -> tuple[bool, str]:

    if not isinstance(action, dict):
        return False, "action_not_a_dict"

    action_type = action.get("action")

    if action_type not in VALID_ACTION_TYPES:
        return False, "unknown_action_type"

    if action_type == "finish":
        return True, "ok"

    if action_type == "back":
        return True, "ok"

    if action_type == "tap":
        target = action.get("target")
        if not target or not isinstance(target, str):
            return False, "missing_target"
        if not is_valid_target(screen, target):
            return False, "target_not_on_screen"
        return True, "ok"

    if action_type == "type":
        target = action.get("target")
        text = action.get("text")
        if not target or not isinstance(target, str):
            return False, "missing_target"
        if text is None or not isinstance(text, str):
            return False, "missing_text"
        if not is_valid_target(screen, target):
            return False, "target_not_on_screen"

        if target != "note_input":
            return False, "target_not_typeable"
        return True, "ok"


    return False, "unknown_action_type"