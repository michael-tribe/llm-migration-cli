from rich.panel import Panel


def create_progress_bar_panel(n_done: int, total: int, width: int) -> str:
    percentage_done = n_done / total
    filled_length = int(width * percentage_done)
    bar = "█" * filled_length + "-" * (width - filled_length)
    percentage_display = f"{percentage_done * 100:.2f}%"
    progress_bar = f"[{bar}] {percentage_display} ({n_done}/{total})"
    return Panel(progress_bar, title="Progress", border_style="green")
