import networkx as nx
import numpy as np
import pandas as pd

# Imports for bqplot
from IPython.display import display
from bqplot import (
    OrdinalScale, LinearScale, Bars, Lines, Axis, Figure, Tooltip
)
from ipywidgets import HBox, VBox, Dropdown, Layout


class GenDispatchBars(Bars):
    """Creates bqplot Bars Mark from solved PSST model.

    Parameters
    ----------
    model : PSSTModel
        The model must have been solved to be used.
    selected_gen : String (Default: all)
        The name of a specific generator you wish to display
    bar_colors : :obj:`list` of :obj:`str` (Default: 'CATEGORY10')
        List of valid HTML colors. Should be one color for each generator.
    padding : Float (Default: 0.2)
        Desired spacing between bars.
    enable_tooltip : Boolean (Default: True)
        Should info be displayed when bars are hovered over
    custom_tooltip: bqpot TooltipModel
        You can create a custom Ipywidgets Tooltip object and pass it as argument

    Attributes
    ----------
    selected_gen : String
        The name of the generator currently being displayed.
    gens : :obj:`list` of :obj:`str`
        List of all generators in model, by name.
    x : numpy.ndarray
        Numpy array of x values
    y : numpy.ndarray, hour of day
        Numpy array of y values, gen ED
    """

    def __init__(self, model, **kwargs):

        # Get data from model
        self._data = model.results.power_generated  # type: Pandas dataframe
        self._gens = list(self._data.columns)  # type: list

        # retrieve keyword args
        self._bar_colors = kwargs.get('bar_colors')
        self._selected_gen = kwargs.get('selected_gen')
        self._enable_tooltip = kwargs.get('enable_tooltip', True)
        self._custom_tooltip = kwargs.get('custom_tooltip')

        # Set and adjust vars for bar chart
        tt = self._make_tooltip()

        self._x_data = range(1, len(self._data)+1)
        self._y_data = []
        for gen in self._gens:
            self._y_data.append(self._data[gen].as_matrix())

        x_sc = OrdinalScale()
        y_sc = LinearScale()

        # Set essential bar chart attributes
        if self._bar_colors:
            self.colors = self._bar_colors
        self._all_colors = self.colors

        self.display_legend=True
        self.padding=0.2

        self.scales={'x': x_sc, 'y':y_sc}
        self.tooltip=tt

        self.x = self._x_data
        self.set_y()

        # Construct bqplot Bars object
        super(GenDispatchBars, self).__init__(**kwargs)


    def set_y(self):
        """ Called by change_selected_gen() and by __init__() to determine
        the appropriate array of data to use for self.y, either all
        generators or just the selected one.
        """

        def _setup_y_all():
            self.color_mode='auto'
            self.labels = self._gens
            self.y=self._y_data
            self.colors = self._all_colors

        def _setup_y_selected(gen_index):
            self.color_mode='element'
            self.labels = [self._gens[gen_index]]
            self.y=self._y_data[gen_index]
            try:
                self.colors = [self._all_colors[gen_index]]
            except IndexError:
                self.colors = self._all_colors

        if self._selected_gen:
            try:
                gen_index = self._gens.index(self._selected_gen)
                _setup_y_selected(gen_index)
            except ValueError:
                warnings.warn("You tried to select non-existstant generator. Displaying all.")
                _setup_y_all()
        else:
            _setup_y_all()


    def _make_tooltip(self):
        """If toolip is true, create it, either with default
        settings or using custom_tooltip object."""
        if self._enable_tooltip:
            if self._custom_tooltip:
                tt = self._custom_tooltip
            else:
                tt = Tooltip(fields=['x', 'y'], formats=['','.2f'],
                             labels=['Hour','ED'])
        else:
            tt = None
        return tt

    @property
    def selected_gen(self):
        return self._selected_gen

    @selected_gen.setter
    def selected_gen(self, gen_name):
        self._selected_gen = gen_name
        self.set_y()

    @property
    def gens(self):
        return self._gens

    @property
    def data(self):
        return self._data

    @property
    def enable_tooltip(self):
        return self._enable_tooltip

    @enable_tooltip.setter
    def enable_tooltip(self, value):
        if value == True:
            self._enable_tooltip = True
        elif value == False:
            self._enable_tooltip = False
        else:
            print("Note: You tried setting enable_tooltip to something "
                  "other than a boolean, so the value did not change.")
        tt = self._make_tooltip()
        self.tooltip = tt

    @property
    def custom_tooltip(self):
        return self._custom_tooltip

    @custom_tooltip.setter
    def custom_tooltip(self, custom_tt):
        self._custom_tooltip = custom_tt
        tt = self._make_tooltip()
        self.tooltip = tt


class GenDispatchFigure(Figure):
    """Creates a bqplot Figure from solved PSST model, containing bars.

    Parameters
    ----------
    model : PSSTModel
        The model must have been solved to be used.
    selected_gen : String (Default: all)
        The name of a specific generator you wish to display
    bar_colors : :obj:`list` of :obj:`str` (Default: 'CATEGORY10')
        List of valid HTML colors. Should be one color for each generator.
    padding : Float (Default: 0.2)
        Desired spacing between bars.
    enable_tooltip : Boolean (Default: True)
        Should info be displayed when bars are hovered over
    custom_tooltip : bqpot TooltipModel
        You can create a custom Ipywidgets Tooltip object and pass it as argument
    x_label : String (Default: 'Hour')
        Label for the figure's x axis
    y_label : String (Default: '')
        Label for the figure's y axis
    additional_marks : :obj:`list` of :obj:`bqplot.Mark`
        Add additional bqplot marks to integrate into the Figure

    Attributes
    ----------
    bars : bqplot Bars Mark
        Can access attribtues like bars.selected_gen, bars.x, and bars.y.
    """

    def __init__(self, model, **kwargs):

        # Make Bars Marks
        self.bars = GenDispatchBars(model, **kwargs)

        # Get additional kwargs
        x_label = kwargs.get('x_label', 'Hour')
        y_label = kwargs.get('y_label', 'Power generated (MW)')
        self.custom_title = kwargs.get('custom_title', None)
        additional_marks = kwargs.get('additional_marks', [])

        # Prepare values
        x_sc = self.bars.scales['x']
        y_sc = self.bars.scales['y']

        ax_x = Axis(scale=x_sc, grid_lines='solid',
                         label=x_label, num_ticks=24)
        ax_y = Axis(scale=y_sc, orientation='vertical',
                         tick_format='0.2f',
                         grid_lines='solid', label=y_label)

        fig_title = self._make_title()

        # Set key attribtues for Figure creation
        self.axes = [ax_x, ax_y]
        self.title = fig_title
        self.marks = [self.bars] + additional_marks
        self.animation_duration = 500

        # Construct Figure object
        super(GenDispatchFigure, self).__init__(**kwargs)

    def _make_title(self):
        selected_gen = self.bars.selected_gen

        if self.custom_title:
            fig_title = self.custom_title
        else:
            fig_title = 'Economic Dispatch for Generators'
            if selected_gen:
                fig_title = fig_title.replace('Generators', selected_gen)
        return fig_title

    def change_selected_gen(self, gen_name):
        self.bars.selected_gen = gen_name
        self.title = self._make_title()


class GenDispatchWidget(VBox):
    """ Make intereactive dispatch plot
    """

    def __init__(self, model, **kwargs):

        # Make Figure with bar chart
        self.figure = GenDispatchFigure(model, **kwargs)

        # Prepare atts
        gens = self.figure.bars.gens
        options = ['All'] + gens

        # Define Dropdown Menu And Callback Fcn
        self.dropdown = Dropdown(
            options=options,
            value=options[0],
            description='Generator:',
        )

        # Setup callback function, for dropdown selection
        def gen_changed(change):
            if change.new == 'All':
                self.figure.change_selected_gen(None)
            else:
                self.figure.change_selected_gen(change.new)

        self.dropdown.observe(gen_changed, 'value')


        super(VBox, self).__init__(children=[self.dropdown, self.figure],
                                   layout=Layout(align_items='center',width='100%',height='100%'),
                                   **kwargs)
