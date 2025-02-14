from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import pandas as pd 
import numpy as np


def add_events(ax,x_axis,events):
    for event in events:
        start = pd.to_datetime(events[event]['start'])
        end = pd.to_datetime(events[event]['end'])
        idx_start = np.cumsum(x_axis < start)[-1]
        
        ax.axvline(idx_start, color='black', linestyle='--', label=f'{event} start')
        ax.text(idx_start, -0.1, f'{event}', rotation=90, va='bottom', ha='left')

def add_events_level(ax,x_axis,events):
    for event in events:
        start = pd.to_datetime(events[event]['start'])
        end = pd.to_datetime(events[event]['end'])
        idx_start = np.cumsum(x_axis < start)[-1]
        idx_end = np.cumsum(x_axis < end)[-1]
        

        level = events[event].get('level', None)
        
        if level is not None:
            ax.axhline(level - 0.5, xmin=idx_start / len(x_axis), xmax=idx_end / len(x_axis), 
                       color='green', linestyle='-', lw=5, alpha=0.5)
        
def highlight_severity(ax, x_axis,events):
    for event_i in events:
        start = pd.to_datetime(events[event_i]['start'])
        end = pd.to_datetime(events[event_i]['end'])
        idx_start = np.cumsum(x_axis < start)[-1]
        idx_end = np.cumsum(x_axis < end)[-1]
        severity = events[event_i].get('severity', None)
        if severity is not None:
            color = 'red' if severity == 'high' else 'yellow'
            # Add rectangle patch above the plot to indicate severity
            box_height = 0.05
            ax.add_patch(Rectangle(
                (idx_start, 1),
                width=idx_end - idx_start,
                height=box_height,
                color=color, 
                alpha=0.5,
                transform=ax.get_xaxis_transform(),
                clip_on=False
            ))
            ax.text((idx_start + idx_end) / 2, -0.1+ box_height, 'severity \n'+severity, ha='center', va='bottom', color='black')

            

def create_date_formatter(column_dates):
    """Return a custom formatter function for datetime labels."""
    def custom_date_format(x, pos=None):
        if 0 <= int(x) < len(column_dates):
            return column_dates[int(x)].strftime('%d-%b-%Y')
        return ''  # Return empty string if out of range
    return FuncFormatter(custom_date_format)
