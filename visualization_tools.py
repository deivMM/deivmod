from matplotlib import colormaps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

def get_gradient_colors(n_colors, cmap='Blues'):
    '''
    The function returns a list of colors that form a gradient of length n_colors.
    '''
    cmap = colormaps[cmap]
    return [cmap(i / (n_colors - 1)) for i in range(n_colors)]

def visualizar_puntos_en_orden(points, cmap = 'Blues', scat= False, ylims = None):
    '''
    Darle un repasito a esta funcion... el nombre no me gusta. Se puede mejorar ??
    '''
    x = points.iloc[:,0].values
    y = points.iloc[:,1].values

    z = np.linspace(0, 1, len(points)) 
    f, ax = plt.subplots(figsize=(6, 6),facecolor='.85')
    if scat:
        lc = ax.scatter(x, y, c=z, cmap=cmap)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(x.min(), x.max()))
        lc.set_array(x)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.autoscale()
    
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_ticks([])  # Opcional: quitar las marcas de la barra de color
    if ylims: ax.set_ylim(ylims)
    plt.show()

def visualizar_pruebas_en_orden(pruebas_dict, varn_1, varn_2,  cmap = 'Blues', scat= False, ylims = None):
    '''
    Darle un repasito a esta funcion... el nombre no me gusta. Se puede mejorar ??
    '''
    colors = get_gradient_colors(len(pruebas_dict), cmap=cmap)
    f, ax = plt.subplots(figsize=(6, 6),facecolor='.85')
    for n, (k, v) in enumerate(pruebas_dict.items()):
        if scat:
            ax.scatter(v[varn_1], v[varn_2], color = colors[n])
        else:
            ax.plot(v[varn_1], v[varn_2], color = colors[n], lw=.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=len(pruebas_dict)))
    sm.set_array([]) 
    cbar = f.colorbar(sm, ax=ax)
    cbar.ax.yaxis.set_ticks([])
    if ylims: ax.set_ylim(ylims)
    plt.show()
