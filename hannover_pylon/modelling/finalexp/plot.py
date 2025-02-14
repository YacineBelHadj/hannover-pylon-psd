from rich.table import Table
from rich.console import Console

def display_damage_vs_healthy_table(scores: dict) -> None:
    """
    Given a dictionary of scores, split it into "damage" and "healthy" based on key prefixes,
    then display them side by side in a Rich table. Any remaining keys (e.g., 'no_event')
    are displayed in a separate table.
    
    Parameters:
        scores (dict): A dictionary where keys are event names and values are scores.
    """
    # Split dictionary into damage and healthy, and others.
    damage_scores = {k: v for k, v in scores.items() if k.startswith("damage")}
    healthy_scores = {k: v for k, v in scores.items() if k.startswith("healthy")}
    others = {k: v for k, v in scores.items() if k not in damage_scores and k not in healthy_scores}

    console = Console()
    
    # Create the main table for damage and healthy side-by-side.
    table = Table(title="Damage vs. Healthy", show_header=True, header_style="bold magenta")
    table.add_column("Damage Event", style="dim", width=20)
    table.add_column("Damage Score", justify="right")
    table.add_column("Healthy Event", style="dim", width=20)
    table.add_column("Healthy Score", justify="right")
    
    # Sort items for a clean display.
    damage_items = sorted(damage_scores.items())
    healthy_items = sorted(healthy_scores.items())
    max_len = max(len(damage_items), len(healthy_items))
    
    # Add rows pairing damage and healthy events.
    for i in range(max_len):
        if i < len(damage_items):
            d_event, d_score = damage_items[i]
            d_event_str = d_event
            d_score_str = f"{d_score:.4f}"
        else:
            d_event_str = ""
            d_score_str = ""
            
        if i < len(healthy_items):
            h_event, h_score = healthy_items[i]
            h_event_str = h_event
            h_score_str = f"{h_score:.4f}"
        else:
            h_event_str = ""
            h_score_str = ""
            
        table.add_row(d_event_str, d_score_str, h_event_str, h_score_str)
    
    console.print(table)
    
    # If there are any remaining keys, print them in a separate table.
    if others:
        other_table = Table(title="Others", show_header=True, header_style="bold magenta")
        other_table.add_column("Event", style="dim", width=20)
        other_table.add_column("Score", justify="right")
        for event_name, score in sorted(others.items()):
            other_table.add_row(event_name, f"{score:.4f}")
        console.print(other_table)
