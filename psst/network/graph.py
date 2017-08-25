"""
For visualizing a PSSTCase as an interactive graph.

Dustin Michels
August 2017
"""

from collections import OrderedDict
import pandas as pd
import networkx as nx
import ipywidgets as ipyw
import traitlets as t
import traittypes as tt
import bqplot as bq
import warnings
import itertools
import time

from psst.case import PSSTCase
from psst.network import PSSTNetwork
from psst.network import create_network
from . import graph_styles as style

class NetworkModel(t.HasTraits):
    """The NetworkModel object is created from a PSSTCase object,
    using the PSSTNetwork object. It contains state information neccesary
    to interactively visualize a network.

    Parameters
    ----------
    case : PSSTCase
        An instance of a PSST case.
    sel_bus : str (Default=None)
        The name of a bus in the case to focus on.

    Attributes
    ----------
    view_buses : :obj:`list` of :obj:`str`
        The names of the buses to be displayed.
    pos : pandas.DataFrame
        All the x,y positions of all the nodes.
    edges : pandas.DataFrame
        All the edges between nodes, and their x,y coordiantes.
    x_edges, y_edges : numpy.Array
        List of coordinates for 'start' and 'end' node, for all edges in network.

    bus_x_vals, bus_y_vals : numpy.Array
        Arrays containing coordinates of buses to be displayed.
    bus_x_edges, bus_y_edges : numpy.Array
        List of coordinates for 'start' and 'end' bus for each edge to be displayed.
    bus_names: :obj:`list` of :obj:`str`
        List of names of buses to be displayed.

    gen_x_vals, gen_y_vals : numpy.Array
        Arrays containing coordinates of generators to be displayed.
    gen_x_edges, gen_y_edges : numpy.Array
        List of coordinates for 'start' and 'end' generator for each edge to be displayed.
    gen_names: :obj:`list` of :obj:`str`
        List of names of generators to be displayed.

    load_x_vals, load_y_vals : numpy.Array
        Arrays containing coordinates of loads to be displayed.
    load_x_edges, load_y_edges : numpy.Array
        List of coordinates for 'start' and 'end' bus for each edge to be displayed.
    load_names: :obj:`list` of :obj:`str`
        List of names of loads to be displayed.

    """

    case = t.Instance(PSSTCase, help='The original PSSTCase')
    network = t.Instance(PSSTNetwork)
    G = t.Instance(nx.Graph)

    sel_bus = t.Unicode(allow_none=True)
    view_buses = t.List(trait=t.Unicode)
    all_pos = tt.DataFrame(help='DF with all x,y positions of nodes.')
    pos = tt.DataFrame(help='DF with x,y positions only for display nodes')
    edges = tt.DataFrame()

    x_edges = tt.Array([])
    y_edges = tt.Array([])

    bus_x_vals = tt.Array([])
    bus_y_vals = tt.Array([])
    bus_names = t.List(trait=t.Unicode)
    bus_x_edges = tt.Array([])
    bus_y_edges = tt.Array([])

    gen_x_vals = tt.Array([])
    gen_y_vals = tt.Array([])
    gen_names = t.List(trait=t.Unicode)
    gen_x_edges = tt.Array([])
    gen_y_edges = tt.Array([])

    load_x_vals = tt.Array([])
    load_y_vals = tt.Array([])
    load_names = t.List(trait=t.Unicode)
    load_x_edges = tt.Array([])
    load_y_edges = tt.Array([])

    x_min_view = t.CFloat()
    x_max_view = t.CFloat()
    y_min_view = t.CFloat()
    y_max_view = t.CFloat()

    _VIEW_OFFSET = 50

    def __init__(self, case, sel_bus=None, *args, **kwargs):
        super(NetworkModel, self).__init__(*args, **kwargs)

        # Store PPSTCase, PSSTNetwork, and networkx.Graph
        self.case = case
        self.network = create_network(case=self.case)
        self.G = self.network.graph

        # Make full pos DF.
        self.all_pos = pd.DataFrame(self.network.positions, index=['x', 'y']).T

        # Make full edges DF, with coordinates
        self.all_edges = pd.DataFrame.from_records(self.G.edges(), columns=['start', 'end'])
        self.all_edges['start_x'] = self.all_edges['start'].map(
            lambda e: self.all_pos.loc[e]['x'])
        self.all_edges['end_x'] = self.all_edges['end'].map(
            lambda e: self.all_pos.loc[e]['x'])
        self.all_edges['start_y'] = self.all_edges['start'].map(
            lambda e: self.all_pos.loc[e]['y'])
        self.all_edges['end_y'] = self.all_edges['end'].map(
            lambda e: self.all_pos.loc[e]['y'])

        # Make df with all edge data
        self.x_edges = [tuple(edge) for edge in self.all_edges[['start_x', 'end_x']].values]
        self.y_edges = [tuple(edge) for edge in self.all_edges[['start_y', 'end_y']].values]

        # Set 'start' and 'end' as index for all_edges df
        self.all_edges.set_index(['start', 'end'], inplace=True)

        # Set 'sel_bus' (this should in turn set other variables)
        self.sel_bus = sel_bus

    @t.observe('sel_bus')
    def _callback_selection_change(self, change):
        self.reset_view_buses()

    def reset_view_buses(self):
        if self.sel_bus is None:
            self.view_buses = list(self.case.bus_name)
        else:
            self.view_buses = [self.sel_bus]

    @t.observe('view_buses')
    def _callback_view_change(self, change):
        self.pos = self.subset_positions()
        self.edges = self.subset_edges()

        ##################
        # Get x,y positions for each layer of node (bus, load, and gen.)
        ##################
        bus_pos = self.pos[self.pos.index.isin(self.case.bus_name)]
        self.bus_x_vals = bus_pos['x']
        self.bus_y_vals = bus_pos['y']
        self.bus_names = list(bus_pos.index)

        gen_pos = self.pos[self.pos.index.isin(self.case.gen_name)]
        self.gen_x_vals = gen_pos['x']
        self.gen_y_vals = gen_pos['y']
        self.gen_names = list(gen_pos.index)

        load_pos = self.pos[self.pos.index.isin(self.case.load_name)]
        self.load_x_vals = load_pos['x']
        self.load_y_vals = load_pos['y']
        self.load_names = list(load_pos.index)

        ##################
        # Get x,y positions for each layer of edge (bus, load, and gen.)
        ##################
        edges = self.edges.reset_index()

        _df = edges.loc[edges.start.isin(self.case.bus_name) & edges.end.isin(self.case.bus_name)]
        self.bus_x_edges = [tuple(edge) for edge in _df[['start_x', 'end_x']].values]
        self.bus_y_edges = [tuple(edge) for edge in _df[['start_y', 'end_y']].values]

        _df = edges.loc[edges.start.isin(self.case.gen_name) | edges.end.isin(self.case.gen_name)]
        self.gen_x_edges = [tuple(edge) for edge in _df[['start_x', 'end_x']].values]
        self.gen_y_edges = [tuple(edge) for edge in _df[['start_y', 'end_y']].values]

        _df = edges.loc[edges.start.isin(self.case.load_name) | edges.end.isin(self.case.load_name)]
        self.load_x_edges = [tuple(edge) for edge in _df[['start_x', 'end_x']].values]
        self.load_y_edges = [tuple(edge) for edge in _df[['start_y', 'end_y']].values]

        ##################
        # Get min and max x,y values that should be viewed.
        ##################
        self.x_max_view = self.pos.x.max() + self._VIEW_OFFSET
        self.x_min_view = self.pos.x.min() - self._VIEW_OFFSET
        self.y_max_view = self.pos.y.max() + self._VIEW_OFFSET * 2
        self.y_min_view = self.pos.y.min() - self._VIEW_OFFSET

    def subset_positions(self):
        """Subset self.all_pos based on view_buses list."""
        nodes = [list(self.G.adj[item].keys()) for item in self.view_buses]
        nodes = set(itertools.chain.from_iterable(nodes))
        nodes.update(self.view_buses)
        return self.all_pos.loc[nodes]

    def subset_edges(self):
        """Subset G.edges based on view_buses list."""
        edge_list_fwd = self.G.edges(nbunch=self.view_buses)
        edge_list_rev = [tuple(reversed(tup)) for tup in edge_list_fwd]
        edges_fwd = self.all_edges.loc[edge_list_fwd].dropna()
        edges_rev = self.all_edges.loc[edge_list_rev].dropna()
        edges = edges_fwd.append(edges_rev)
        edges.unstack()
        return edges

    def __repr__(self):
        s = ('ExploreNetworkModel Object\n'
             'sel_bus={}; view_buses={}').format(self.sel_bus, self.view_buses)
        return s


class NetworkViewBase(ipyw.VBox):
    """An interactive, navigable display of the network.

    The NetworkViewBase class simply generates an interactive figure,
    without ipywidget buttons and dropdown menus. The NetworkModel extends
    this class, adding widget controls.

    Parameters
    ----------
    case : PSSTCase
        An instance of a PSST case.
    model : NetworkModel
        An instance of NetworkModel can be passed instead
        of a case. (Should not pass both.)

    Attributes
    ----------
    model : NetworkModel
        An instance of NetworkModel, containing state information for network.
    show_gen : Bool
        Display the points representing generators and connected lines.
    show_load : Bool
        Display the points representing loads and connected lines.
    show_bus_names : Bool
        Display names next to buses.
    show_gen_names : Bool
        Display names next to generators.
    show_load_names : Bool
        Display names next to loads.
    """

    model = t.Instance(NetworkModel)
    show_gen = t.Bool(default_value=True)
    show_load = t.Bool(default_value=True)

    show_background_lines = t.Bool(default_value=True)
    show_bus_names = t.Bool(default_value=True)
    show_gen_names = t.Bool(default_value=True)
    show_load_names = t.Bool(default_value=True)

    def __init__(self, case=None, model=None, *args, **kwargs):
        super(NetworkViewBase, self).__init__(*args, **kwargs)

        ##################
        # Load and Store Model
        ##################
        if model and case:
            warnings.warn('You should pass a case OR a model, not both. The case argument you passed is being ignored.')
        if not model:
            self.model = NetworkModel(case)
        else:
            self.model = model

        ##################
        # Scale Marks
        ##################
        self._scale_x = bq.LinearScale(
            min=self.model.x_min_view,
            max=self.model.x_max_view
        )
        self._scale_y = bq.LinearScale(
            min=self.model.y_min_view,
            max=self.model.y_max_view
        )
        self._scales = {
            'x': self._scale_x,
            'y': self._scale_y,
        }

        ##################
        # Scatter Marks
        ##################
        # Tooltip
        scatter_tooltip = bq.Tooltip(fields=['name'])

        # Create Bus Scatter
        self._bus_scatter = bq.Scatter(
            x=self.model.bus_x_vals, y=self.model.bus_y_vals,
            scales=self._scales, names=self.model.bus_names,
            marker='rectangle', default_size=180,
            colors=[style.graph_main_1], default_opacities=[0.6],
            selected_style={'opacity': 0.8, 'fill': style.graph_selected_1,
                            'stroke':style.graph_accent_1},
            selected=self._get_indices_view_buses(),
            tooltip=scatter_tooltip,
        )
        # Create Gen Scatter
        self._gen_scatter = bq.Scatter(
            x=self.model.gen_x_vals, y=self.model.gen_y_vals,
            scales=self._scales, names=self.model.gen_names,
            marker='circle', default_size=150,
            colors=[style.graph_main_2], default_opacities=[0.6],
            selected_style={'opacity': 0.8, 'fill': style.graph_selected_2,
                            'stroke': style.graph_accent_2},
            tooltip=scatter_tooltip,
        )
        # Create Load Scatter
        self._load_scatter = bq.Scatter(
            x=self.model.load_x_vals, y=self.model.load_y_vals,
            scales=self._scales, names=self.model.load_names,
            marker='triangle-up', default_size=140,
            colors=[style.graph_main_3], default_opacities=[0.6],
            selected_style={'opacity': 0.8, 'fill': style.graph_selected_3,
                            'stroke': style.graph_accent_3},
            tooltip=scatter_tooltip,
        )

        ##################
        # Line Marks
        ##################
        # Create Bus Lines
        self._bus_lines = bq.Lines(
            x=self.model.bus_x_edges, y=self.model.bus_y_edges,
            scales=self._scales, colors=[style.graph_line_1],
            stroke_width=2, line_style='solid',
            opacities=[0.8],
        )
        # Create Gen Lines
        self._gen_lines = bq.Lines(
            x=self.model.gen_x_edges, y=self.model.gen_y_edges,
            scales=self._scales, colors=[style.graph_line_2],
            stroke_width=1.5, line_style='solid',
            opacities=[0.8]
        )
        # Create Load Lines
        self._load_lines = bq.Lines(
            x=self.model.load_x_edges, y=self.model.load_y_edges,
            scales=self._scales, colors=[style.graph_line_3],
            stroke_width=1.5, line_style='solid',
            opacities=[0.8],
        )
        # All lines, in background
        self._background_lines = bq.Lines(
            x=self.model.x_edges, y=self.model.y_edges,
            scales=self._scales, colors=['gray'],
            stroke_width=1, line_style='dotted',
            opacities=[0.05], marker='circle', marker_size=30,
        )

        ##################
        # Bqplot Figure
        ##################
        self._all_marks = OrderedDict({
            'background_lines': self._background_lines,
            'bus_lines': self._bus_lines,
            'gen_lines': self._gen_lines,
            'load_lines': self._load_lines,
            'bus_scatter': self._bus_scatter,
            'gen_scatter': self._gen_scatter,
            'load_scatter': self._load_scatter,
        })

        fig_margin = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        self._figure = bq.Figure(
            marks=list(self._all_marks.values()),
            animation_duration=0,
            fig_margin=fig_margin,
        )

        # Set as children of VBox
        self.children = [self._figure]

        # Set defaults, triggering callback functions (setting proper children)
        self.show_background_lines = False
        self.show_gen_names = False
        self.show_load_names = False

        ##################
        # Link Traits
        ##################
        # Link Scales
        t.link((self.model, 'x_min_view'), (self._scale_x, 'min'))
        t.link((self.model, 'x_max_view'), (self._scale_x, 'max'))
        t.link((self.model, 'y_min_view'), (self._scale_y, 'min'))
        t.link((self.model, 'y_max_view'), (self._scale_y, 'max'))

        # Link Bus Scatter
        t.link((self.model, 'bus_x_vals'), (self._bus_scatter, 'x'))
        t.link((self.model, 'bus_y_vals'), (self._bus_scatter, 'y'))
        t.link((self.model, 'bus_names'), (self._bus_scatter, 'names'))

        # Link Gen Scatter
        t.link((self.model, 'gen_x_vals'), (self._gen_scatter, 'x'))
        t.link((self.model, 'gen_y_vals'), (self._gen_scatter, 'y'))
        t.link((self.model, 'gen_names'), (self._gen_scatter, 'names'))

        # Link Load Scatter
        t.link((self.model, 'load_x_vals'), (self._load_scatter, 'x'))
        t.link((self.model, 'load_y_vals'), (self._load_scatter, 'y'))
        t.link((self.model, 'load_names'), (self._load_scatter, 'names'))

        # Link Bus Lines
        t.link((self.model, 'bus_x_edges'), (self._bus_lines, 'x'))
        t.link((self.model, 'bus_y_edges'), (self._bus_lines, 'y'))

        # Link Gen Lines
        t.link((self.model, 'gen_x_edges'), (self._gen_lines, 'x'))
        t.link((self.model, 'gen_y_edges'), (self._gen_lines, 'y'))

        # Link Load Lines
        t.link((self.model, 'load_x_edges'), (self._load_lines, 'x'))
        t.link((self.model, 'load_y_edges'), (self._load_lines, 'y'))

        # Link names to show
        t.link((self, 'show_bus_names'), (self._bus_scatter, 'display_names'))
        t.link((self, 'show_gen_names'), (self._gen_scatter, 'display_names'))
        t.link((self, 'show_load_names'), (self._load_scatter, 'display_names'))

        # Set callbacks for clicking a bus
        # Click -> updates `model.view_buses` -> updates `bus_scatter.selected`
        self._bus_scatter.on_element_click(self._callback_bus_clicked)
        self.model.observe(self._callback_view_buses_change, names='view_buses')

        # Callbacks for clicking a load/gen node (Simply flashes selected)
        self._gen_scatter.on_element_click(self._callback_nonebus_clicked)
        self._load_scatter.on_element_click(self._callback_nonebus_clicked)

    def _get_indices_view_buses(self):
        """Returns names in model.view_buses as a list of corresponding indices."""
        idx_list = [self.model.bus_names.index(bus) for bus in self.model.view_buses]
        return idx_list

    def _callback_bus_clicked(self, scatter, change):
        """When a bus is clicked, add/remove it from `model.view_buses.'
        If the last view_bus is clicked, flash red to indicate can't click."""
        name = change['data']['name']
        new_list = self.model.view_buses[:]
        if [name] != self.model.view_buses:  # Not the last bus
            if name in new_list:
                new_list.remove(name)
            else:
                new_list.append(name)
            self.model.view_buses = list(set(new_list))
        else:
            tmp = scatter.stroke
            scatter.stroke = 'red'
            time.sleep(.2)
            scatter.stroke = tmp

    def _callback_view_buses_change(self, change):
        """When `model.view_buses` changes, update `selected` buses in plot."""
        idx_list = self._get_indices_view_buses()
        self._bus_scatter.selected = idx_list

    def _callback_nonebus_clicked(self, scatter, change):
        """When a load or generator is clicked, have it flash selected."""
        scatter.selected = [change['data']['index']]
        time.sleep(.2)
        scatter.selected = []

    @t.observe('show_gen', 'show_load', 'show_background_lines')
    def _callback_update_shown(self, change):
        """When a bool property that effects which marks to display is changed,
        check the values of all such properties and build the appropriate mark list."""
        mark_names = list(self._all_marks)  # Get just the names (keys)
        if self.show_gen is False:
            mark_names.remove('gen_scatter')
            mark_names.remove('gen_lines')
        if self.show_load is False:
            mark_names.remove('load_scatter')
            mark_names.remove('load_lines')
        if self.show_background_lines is False:
            mark_names.remove('background_lines')
        new_marks = [self._all_marks[k] for k in mark_names]
        self._figure.marks = new_marks

    def reset_view(self):
        """Reset the displayed buses to reflect `model.sel_bus`,
        undoing any changes caused by clicking."""
        self.model.reset_view_buses()


class NetworkView(NetworkViewBase):
    """Create an interactive, navigable display of the network, with widget controls.

    This class extends NetworkViewBase, adding ipywidget buttons, dropdowns,
    and checkboxes on top of the core, interactive network graph.

    Parameters
    ----------
    case : PSSTCase
        An instance of a PSST case.
    model : NetworkModel
        An instance of NetworkModel can be passed instead
        of a case. (Should not pass both.)

    Attributes
    ----------
    model : NetworkModel
        An instance of NetworkModel, containing state information for network.
    show_gen : Bool
        Display the points representing generators and connected lines.
    show_load : Bool
        Display the points representing loads and connected lines.
    show_bus_names : Bool
        Display names next to buses.
    show_gen_names : Bool
        Display names next to generators.
    show_load_names : Bool
        Display names next to loads.
    """

    def __init__(self, case=None, model=None, *args, **kwargs):
        super(NetworkView, self).__init__(case, model, *args, **kwargs)

        ##################
        # Additional widgets
        ##################

        self._selector = ipyw.Dropdown(
            description="Start at:",
            options=["All"] + list(self.model.case.bus_name)
        )
        self._reset = ipyw.Button(
            description='Reset',
            button_style='info',
            icon='times-circle'
        )
        self._show_gen_checkbox = ipyw.Checkbox(
            value=True, description="Show generators", indent=False,
        )
        self._show_load_checkbox = ipyw.Checkbox(
            value=True, description="Show loads", indent=False,
        )
        self._show_bus_names_checkbox = ipyw.Checkbox(
            value=True, description="Show bus names", indent=False,
        )
        self._show_gen_names_checkbox = ipyw.Checkbox(
            value=True, description="Show gen names", indent=False,
        )
        self._show_load_names_checkbox = ipyw.Checkbox(
            value=True, description="Show load names", indent=False,
        )
        self._show_background_lines_checkbox = ipyw.Checkbox(
            value=False, description="Show hidden nodes", indent=False,
        )

        ##################
        # Children Layouts & Arrangement
        ##################
        # box_topbar (For selecting starting bus)
        topbar_items = [
            ipyw.HBox([self._selector, self._reset]),
            ipyw.Label('Click on a bus to expand the graph, and click again to collapse.')]
        box_topbar = ipyw.VBox(children=topbar_items, layout=style.topbar_layout)

        # box_sidebar (For adjusting options)
        sidebar_items = [
            ipyw.Label("Options:"),
            ipyw.Label('--------------------'),
            self._show_gen_checkbox,
            self._show_load_checkbox,
            ipyw.Label('--------------------'),
            self._show_bus_names_checkbox,
            self._show_gen_names_checkbox,
            self._show_load_names_checkbox,
            ipyw.Label('--------------------'),
            self._show_background_lines_checkbox]
        [setattr(item, 'layout', style.sidebar_item_layout) for item in sidebar_items]
        box_sidebar = ipyw.VBox(children=sidebar_items, layout=style.sidebar_layout)

        # box_body: Options sidebar + figure
        body_items = [
            box_sidebar, self._figure]
        self._figure.layout = style.figure_layout
        box_body = ipyw.HBox(body_items, layout=style.body_layout)

        # Overall View
        children = [box_topbar, box_body]
        self.children = children
        self.layout.max_width = '810px'

        ##################
        # Add Links
        ##################
        # Link Selector
        t.dlink(source=(self.model, 'sel_bus'),
                target=(self._selector, 'value'),
                transform=(lambda x: "All" if x is None else x))
        t.dlink(source=(self._selector, 'value'),
                target=(self.model, 'sel_bus'),
                transform=(lambda x: None if x is "All" else x))

        # Link checkboxes
        t.link((self, 'show_gen'), (self._show_gen_checkbox, 'value'))
        t.link((self, 'show_load'), (self._show_load_checkbox, 'value'))
        t.link((self, 'show_background_lines'), (self._show_background_lines_checkbox, 'value'))
        t.link((self, 'show_bus_names'), (self._show_bus_names_checkbox, 'value'))
        t.link((self, 'show_gen_names'), (self._show_gen_names_checkbox, 'value'))
        t.link((self, 'show_load_names'), (self._show_load_names_checkbox, 'value'))

        # Callback for reset button press.
        self._reset.on_click(self._callback_reset_press)

    def _callback_reset_press(self, change):
        """When reset button is pressed, call reset_view,
        undoing changes to `model.view_buses` made by clicking."""
        self.reset_view()
