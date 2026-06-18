import re

from mobile_ui_env.dataset import build_dataset
from mobile_ui_env.env import MobileUIEnv
from mobile_ui_env.state import NOTES, SETTINGS, PROFILE


# Rule-based baseline agent: instruction (text) -> action sequence


def _extract_quoted_title(instruction: str) -> str | None:
    """Pull a single-quoted substring out of the instruction, e.g.
    "Create a note titled 'Buy milk'" -> "Buy milk". Returns None if no
    quoted text is found."""
    match = re.search(r"'([^']+)'", instruction)
    return match.group(1) if match else None


def plan_actions(instruction: str) -> list[dict]:

    text = instruction.lower()

    # --- two notes (check before single-note, since both mention "note") ---
    if "two" in text and "note" in text:
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "First note"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Second note"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]

    # --- single note creation ---
    if "note" in text and ("create" in text or "titled" in text):
        title = _extract_quoted_title(instruction) or "Untitled note"
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": title},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]

    # --- focus mode ---
    if "focus mode" in text:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"},
        ]

    # --- notifications ---
    if "notification" in text:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"},
        ]

    # --- app version ---
    if "version" in text:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "version_label"},
            {"action": "finish"},
        ]

   
    if "profile" in text and ("not log out" in text or "without logout" in text or "do not logout" in text):
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"},
        ]

    # --- username ---
    if "username" in text:
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "username_label"},
            {"action": "finish"},
        ]

    # --- email ---
    if "email" in text:
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "email_label"},
            {"action": "finish"},
        ]

    # --- generic "navigate to X screen" ---
    if "navigate" in text or "go to" in text:
        if NOTES in text:
            return [{"action": "tap", "target": "notes_button"}, {"action": "finish"}]
        if SETTINGS in text:
            return [{"action": "tap", "target": "settings_button"}, {"action": "finish"}]
        if PROFILE in text:
            return [{"action": "tap", "target": "profile_button"}, {"action": "finish"}]


    return [{"action": "finish"}]


# Eval loop


def run_eval() -> None:
    eval_tasks = build_dataset(split="eval")

    total = len(eval_tasks)
    successes = 0
    rewards = []
    steps = []
    invalid_counts = []
    safety_counts = []

    print(f"Running baseline agent on {total} eval tasks...\n")

    for task in eval_tasks:
        env = MobileUIEnv(task)
        actions = plan_actions(task["instruction"])
        obs, reward, done, info = env.run_episode(actions)

        from mobile_ui_env.rubric import check_goal
        success = check_goal(env.state, task["goal"])

        successes += int(success)
        rewards.append(reward)
        steps.append(info["step_count"])
        invalid_counts.append(info["invalid_action_count"])
        safety_counts.append(info["safety_violations"])

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {task['task_id']:<28} reward={reward:.2f}  steps={info['step_count']}")

    success_rate = successes / total
    avg_reward = sum(rewards) / total
    avg_steps = sum(steps) / total
    total_actions = sum(steps)
    invalid_rate = sum(invalid_counts) / total_actions if total_actions > 0 else 0.0
    total_safety_violations = sum(safety_counts)

    print()
    print(f"Total eval tasks: {total}")
    print(f"Success rate: {success_rate * 100:.0f}%")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Invalid action rate: {invalid_rate:.2f}")
    print(f"Safety violations: {total_safety_violations}")

    if success_rate == 1.0:
        print(
            "\nNote: 100% success here reflects that this keyword-based "
            "baseline's rules and the dataset's instruction phrasing were "
            "written in the same pass, so keywords line up by construction. "
            "It is not evidence the agent generalizes to novel phrasing -- "
            "e.g. 'jot down a reminder' (no 'note'/'create'/'titled' "
            "keyword) fails and falls back to an immediate finish. Treat "
            "this baseline as a sanity check that the environment, dataset, "
            "and reward pipeline work end-to-end, not as a capability claim."
        )


if __name__ == "__main__":
    run_eval()