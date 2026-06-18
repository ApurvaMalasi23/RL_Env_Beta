from .state import HOME, NOTES, SETTINGS, PROFILE


def _task(task_id: str, instruction: str, goal: dict, max_steps: int = 8) -> dict:
    """Small constructor to keep every task dict shaped identically."""
    return {
        "task_id": task_id,
        "instruction": instruction,
        "goal": goal,
        "max_steps": max_steps,
    }

# Train tasks (20)


def _build_train_tasks() -> list[dict]:
    tasks = []

    # create note (5 variants, different titles)
    titles = ["Buy milk", "Call mom", "Finish report", "Book flight", "Water plants"]
    for i, title in enumerate(titles, start=1):
        tasks.append(
            _task(
                f"train_note_{i:02d}",
                f"Create a note titled '{title}'",
                {"type": "note_created", "title": title},
            )
        )

    # create two notes (2 variants)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_two_notes_{i:02d}",
                "Create two separate notes",
                {"type": "two_notes_created"},
                max_steps=12,
            )
        )

    # enable focus mode (2)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_focus_on_{i:02d}",
                "Enable focus mode in settings",
                {"type": "focus_mode_enabled"},
            )
        )

    # disable notifications (2)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_notif_off_{i:02d}",
                "Disable notifications in settings",
                {"type": "notifications_disabled"},
            )
        )

    # report username (2)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_username_{i:02d}",
                "Go to your profile and find your username",
                {"type": "username_viewed"},
            )
        )

    # report email (2)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_email_{i:02d}",
                "Go to your profile and find your email address",
                {"type": "email_viewed"},
            )
        )

    # report version (2)
    for i in range(1, 3):
        tasks.append(
            _task(
                f"train_version_{i:02d}",
                "Open settings and report the app version",
                {"type": "version_viewed"},
            )
        )

    # navigate to screen (2, different target screens)
    for i, screen in enumerate([NOTES, SETTINGS], start=1):
        tasks.append(
            _task(
                f"train_navigate_{i:02d}",
                f"Navigate to the {screen} screen",
                {"type": "screen_visited", "screen": screen},
                max_steps=4,
            )
        )

    # visit profile without logout (1)
    tasks.append(
        _task(
            "train_profile_no_logout_01",
            "Visit your profile but do not log out",
            {"type": "profile_no_logout"},
        )
    )

    return tasks

# Eval tasks (10) -- different concrete params than train, same goal types


def _build_eval_tasks() -> list[dict]:
    tasks = []

    eval_titles = ["Pay rent", "Pack bags"]
    for i, title in enumerate(eval_titles, start=1):
        tasks.append(
            _task(
                f"eval_note_{i:02d}",
                f"Create a note titled '{title}'",
                {"type": "note_created", "title": title},
            )
        )

    tasks.append(
        _task(
            "eval_two_notes_01",
            "Create two separate notes",
            {"type": "two_notes_created"},
            max_steps=12,
        )
    )

    tasks.append(
        _task(
            "eval_focus_on_01",
            "Enable focus mode in settings",
            {"type": "focus_mode_enabled"},
        )
    )

    tasks.append(
        _task(
            "eval_notif_off_01",
            "Disable notifications in settings",
            {"type": "notifications_disabled"},
        )
    )

    tasks.append(
        _task(
            "eval_username_01",
            "Go to your profile and find your username",
            {"type": "username_viewed"},
        )
    )

    tasks.append(
        _task(
            "eval_email_01",
            "Go to your profile and find your email address",
            {"type": "email_viewed"},
        )
    )

    tasks.append(
        _task(
            "eval_version_01",
            "Open settings and report the app version",
            {"type": "version_viewed"},
        )
    )

    tasks.append(
        _task(
            "eval_navigate_01",
            f"Navigate to the {PROFILE} screen",
            {"type": "screen_visited", "screen": PROFILE},
            max_steps=4,
        )
    )

    tasks.append(
        _task(
            "eval_profile_no_logout_01",
            "Visit your profile but do not log out",
            {"type": "profile_no_logout"},
        )
    )

    return tasks


_TRAIN_TASKS = _build_train_tasks()
_EVAL_TASKS = _build_eval_tasks()


def build_dataset(split: str = "train") -> list[dict]:
    """
    Return the list of task dicts for the requested split.

    split: "train" (20 tasks) or "eval" (10 tasks).
    """
    if split == "train":
        return list(_TRAIN_TASKS)
    if split == "eval":
        return list(_EVAL_TASKS)
    raise ValueError(f"Unknown split '{split}', expected 'train' or 'eval'")