"""
This module attempts to abstract and consolidate all
the color and styling information for the graph object
into one, easily editable place.
"""

import psst.colors as colors
import ipywidgets as ipyw

##################
# Colors
##################
graph_main_1 = colors.blue1
graph_selected_1 = colors.blue2
graph_accent_1 = colors.blue3

graph_main_2 = colors.green1
graph_selected_2 = colors.green2
graph_accent_2 = colors.green3

graph_main_3 = colors.orange1
graph_selected_3 = colors.orange2
graph_accent_3 = colors.orange3

graph_line_1 = colors.blue1
graph_line_2 = colors.green1
graph_line_3 = colors.orange1

graph_section_border = 'Blue'
graph_subsection_border = 'Gray'


##################
# Layouts
##################
section_border = 'groove {} 2px'.format(graph_section_border)
subsection_border = 'dotted {} 1px'.format(graph_subsection_border)

# Topbar: Dropdown Selector + Reset Button
topbar_layout = ipyw.Layout(
    border=section_border, align_items='center'
)
# Sidebar: Options checkboxes
sidebar_layout = ipyw.Layout(
    min_width='150px', margin='5px',
    border=subsection_border
)
sidebar_item_layout = ipyw.Layout(
    width='auto'
)
# Figure: Scatter and Lines Marks comprising graph
figure_layout = ipyw.Layout(
    margin='5px', border=subsection_border
)
# Body: Sidebar + Figure
body_layout = ipyw.Layout(
    justify_content='space-between',
    border=section_border, height='auto'
)