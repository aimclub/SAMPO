from enum import Flag, auto
from typing import Optional, Union

from matplotlib.pyplot import Figure as MatplotlibFig
from plotly.graph_objects import Figure as PlotlyFig


class VisualizationMode(Flag):
    NoFig = 0
    ShowFig = auto()
    SaveFig = auto()
    ReturnFig = auto()


def visualize(fig: Union[PlotlyFig, MatplotlibFig], mode: VisualizationMode, file_name: str = None) \
        -> Optional[Union[PlotlyFig, MatplotlibFig]]:
    """
    Visualizes the figure according to the provided settings
    :param fig: The figure
    :param mode: Visualisation mode. Can be bitwise-or (|) of several modes.
    :param file_name: Optional name of a saved file. Passed, if SaveFig in mode.
    :return: The figure, if ReturnFig in mode. Otherwise, None.
    """
    if VisualizationMode.SaveFig in mode:
        if isinstance(fig, PlotlyFig):
            fig.write_image(file_name)
        else:
            fig.savefig(file_name)
    if VisualizationMode.ShowFig in mode:
        fig.show()
    if VisualizationMode.ReturnFig in mode:
        return fig
    return None
