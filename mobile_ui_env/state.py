from dataclasses import dataclass, field

HOME = "home"
NOTES = "notes"
SETTINGS = "settings"
PROFILE = "profile"

ALL_SCREENS = (HOME, NOTES, SETTINGS, PROFILE)


@dataclass
class AppState:
    # --- Navigation ---
    current_screen: str = HOME

    # --- Notes screen state ---
    notes: list[str] = field(default_factory=list)
    note_input_text: str = ""

    # --- Settings screen state ---
    focus_mode: bool = False
    notifications: bool = True  

    # --- Profile screen state ---
    username: str = "alex_dev"
    email: str = "alex_dev@example.com"
    app_version: str = "1.4.2"

  
    viewed_username: bool = False
    viewed_email: bool = False
    viewed_version: bool = False

    # --- Safety-relevant state ---
    logged_out: bool = False

    step_count: int = 0
    invalid_action_count: int = 0
    safety_violations: int = 0
    done: bool = False