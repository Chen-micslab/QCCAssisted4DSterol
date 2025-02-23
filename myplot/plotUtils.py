import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Iterable, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib.transforms as mtransforms
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, \
    median_absolute_error, r2_score


def x_vs_y_fitting(x: object, y: object, data: pd.DataFrame = None, ax=None, fig=None, grid: bool = False,
                   figsize: tuple = (6.5, 6.5), plot_xy: bool = True, use_fitting: bool = False,
                   plot_arg=None, line_style: Union[str, tuple] = None,
                   scatter_arg: dict = None, scatter_size: float = None,
                   grid_arg: dict = None,
                   text_shift: tuple = (10, -30), text_size: float = 6.0,
                   xy_label_size: float = 6.0, x_label: str = 'x', y_label: str = 'y',
                   xy_tick_size: float = 6.0, major_locator: Tuple[float, float] = None,
                   unit: str = None, title: str = None, title_size: float = 6.0,labelpad:float=10,
                   ):
    if plot_arg is None:
        plot_arg = {'linewidth': 2, 'color': 'black', 'linestyle': (0, (2, 2))}
    if scatter_arg is None:
        scatter_arg = {'s': 20, 'c': myColorList3[0]}

    if line_style:
        plot_arg['linestyle'] = line_style

    x_data = None
    y_data = None
    data_type = 0
    if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
        try:
            x_data = np.array(x)
            y_data = np.array(y)
            data_type = 1
        except:
            pass
    if isinstance(data, pd.DataFrame) and data_type == 0:
        if isinstance(x, str) and isinstance(y, str):
            try:
                x_data = data[x].values
                y_data = data[y].values
            except:
                pass
    if not (isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray)):
        raise Exception('Data type wrong')
    ax_ = None
    if ax:
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(1, 1, figsize=cm_2_inch(figsize[0], figsize[1]))
    if grid:
        if grid_arg:
            ax_.grid(**grid_arg, zorder=0)
        else:
            ax_.grid(True, **{'color': '.8', 'linestyle': (0, (2, 2))}, zorder=0)
    if scatter_arg:
        if scatter_size:
            scatter_arg['s'] = scatter_size
            ax_.scatter(x_data, y_data, **scatter_arg)
    else:
        ax_.scatter(x_data, y_data)
    x_min, x_max = ax_.get_xlim()
    plot_data_x = np.arange(x_min * 0.9, x_max * 1.1, 0.001)
    text_formula = None
    if use_fitting:
        k, y = np.polyfit(x_data, y_data, 1)
        plot_data_y = k * plot_data_x + y
        if plot_xy:
            if plot_arg:
                ax_.plot(plot_data_x, plot_data_y, **plot_arg)
            else:
                ax_.plot(plot_data_x, plot_data_y)
        if y >= 0:
            text_formula = f'y = {str(round(k, 3))}x + {str(round(y, 3))}'
        else:
            y = abs(y)
            text_formula = f'y = {str(round(k, 3))}x - {str(round(y, 3))}'
    else:
        plot_data_y = plot_data_x
        if plot_xy:
            if plot_arg:
                ax_.plot(plot_data_x, plot_data_y, **plot_arg)
            else:
                ax_.plot(plot_data_x, plot_data_y)
    MedRE = np.median(abs(x_data - y_data) / x_data)
    MeanRE = np.mean(abs(x_data - y_data) / x_data)
    correlation_coefficient = np.corrcoef(x_data, y_data)[0, 1]
    r2 = r2_score(x_data, y_data)
    text_r2 = f'R\u00b2 = {r2:.3f}'
    text_MeanRE = f'MRE: {MeanRE:.2%}'
    text_MedRE = f'MdMRE: {MedRE:.2%}'
    if text_formula:
        text = text_formula + '\n' + text_r2 + '\n' + text_MeanRE + '\n' + text_MedRE
    else:
        text = text_r2 + '\n' + text_MeanRE + '\n' + text_MedRE
    trans = mtransforms.ScaledTranslation(text_shift[0] / 72, text_shift[1] / 72, fig.dpi_scale_trans)
    ax_.text(0.0, 1.0, text, transform=ax_.transAxes + trans, fontdict=fontdict_lable_generate(text_size))
    plt.setp(ax_.get_xticklabels(), **fontdict_lable_generate(xy_tick_size))
    plt.setp(ax_.get_yticklabels(), **fontdict_lable_generate(xy_tick_size))
    if not major_locator:
        ax_.xaxis.set_major_locator(ticker.AutoLocator())
        ax_.yaxis.set_major_locator(ticker.AutoLocator())
    else:
        ax_.xaxis.set_major_locator(ticker.MultipleLocator(major_locator[0]))
        ax_.yaxis.set_major_locator(ticker.MultipleLocator(major_locator[1]))
    if title:
        ax.set_title(title, **fontdict_lable_generate(title_size))
    if unit:
        ax_.set_xlabel(f'{x_label} ({unit})', fontdict=fontdict_lable_generate(xy_label_size), labelpad=labelpad)
        ax_.set_ylabel(f'{y_label} ({unit})', fontdict=fontdict_lable_generate(xy_label_size), labelpad=labelpad)
    else:
        ax_.set_xlabel(f'{x_label}', fontdict=fontdict_lable_generate(xy_label_size), labelpad=labelpad)
        ax_.set_ylabel(f'{y_label}', fontdict=fontdict_lable_generate(xy_label_size), labelpad=labelpad)
    ax.set_axisbelow(True)


def single_pie(x, label, ax=None, figsize: tuple = (6.5, 6.5), colors=None, radius:float=1,
               autopct=None, autofuc=None, title=None, title_size:float=6, pctdistance:float=0.6,
               autotext_kwg:dict= {'size':8, 'family':'Arial', 'color':'black'},
               startangle:float=0.0):
    if ax:
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(1, 1, figsize=cm_2_inch(figsize[0], figsize[1]))
    if autofuc:
        def func(pct, allvals):
            absolute = int(np.round(pct / 100. * np.sum(allvals)))
            # return f"{pct:.1f}%\n{absolute}"
            return f"{pct:.1f}%"
            # return f"{pct:.1f}%\nn={absolute}"


    # def func(pct, allvals):
    #     absolute = int(np.round(pct / 100. * np.sum(allvals)))
    #     return f"{pct:.1f}%"

        wedges, texts, autotexts = ax.pie(x, labels=label, autopct=lambda pct: func(pct, x),pctdistance=pctdistance,
                                          colors=colors, radius=radius,startangle=startangle)
        plt.setp(autotexts, **autotext_kwg)
    else:
        wedges, texts = ax.pie(x, labels=label, pctdistance=pctdistance,
                                          colors=colors, radius=radius,startangle=startangle)

    ax_.set_xlabel(title, fontdict=fontdict_lable_generate(8))


    plt.setp(texts, size=8, family='Arial')
    if title:
        plt.title(title, **fontdict_lable_generate(title_size))


def rgb_int_2_hex(rgb: tuple):
    return f'#{rgb[0]:x}{rgb[1]:x}{rgb[2]:x}'.upper()


def rgb_hex_2_int(rgbStr: str):
    return int(rgbStr[1:3], 16), int(rgbStr[3:5], 16), int(rgbStr[5:7], 16)


def cm_2_inch(x, y):
    return x / 2.54, y / 2.54


def fontdict_lable_generate(fontsize: float = 10, fontname: str = 'Arial',fontweight:str = 'normal'):
    return {'name': fontname, 'fontsize': fontsize, 'fontweight': fontweight}


myColorDict1 = {
    'b': "#3FA9F5", 'r': "#FF1D25", 'g': '#7AC943', 'br': '#FF931E', 'p': '#7030A0'
}
myColorList1 = list(myColorDict1.values())

myColorListRGB = [(235, 29, 37), (245, 176, 65), (246, 218, 101), (82, 190, 128), (145, 223, 208), (93, 173, 226),
                  (164, 105, 189), (138, 112, 103), (255, 188, 167), (72, 79, 152), (255, 255, 133)]

myColorList2 = [rgb_int_2_hex(x) for x in myColorListRGB]

myColorList3 = []
for x in [6, 1, 2, 3, 7, 8, 10]:
    myColorList3.append(myColorList2[x - 1])

myColorPalette = LinearSegmentedColormap('myColor', myColorList1)
mpl.colormaps.register(myColorPalette)
