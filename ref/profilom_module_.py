import socket
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from pickle import TRUE
from scipy.interpolate import interp1d
import math
from scipy import linalg
import pandas as pd
from scipy.spatial.distance import cdist
import time
from datetime import timedelta
from ctypes import*
import platform
from scipy.interpolate import pchip_interpolate
import function_library as flib
import time
from scipy.optimize import minimize
import geom_calc_library as gcalc


def projection_onto_line(point, line_point1, line_point2):
    '''
    Also valid for a vector with 3 coordinates 
    Eg.: projection_onto_line(np.array([1, 2]), np.array([0, 0]), np.array([1, 1]))
    '''
    # Calculate the direction vector of the line
    line_direction = line_point2 - line_point1
    # Calculate vector from line_point1 to the given point
    u = point - line_point1
    # Calculate projection of u onto the line direction vector
    projection = np.dot(u, line_direction) / np.dot(line_direction, line_direction) * line_direction
    # Calculate the projection point
    projection_point = line_point1 + projection
    return projection_point

def midpoint(point1, point2):
    '''
    Also valid for a vector with 3 coordinates 
    Eg.: projection_onto_line(np.array([0, 0]), np.array([1, 10]))
    '''
    midpoint = (point1 + point2) / 2
    return midpoint

def pts_distance(point1, point2):
    '''
    Also valid for a vector with 3 coordinates 
    Eg.: distance(np.array([0, 0]), np.array([1, 10]))
    '''
    return np.sqrt(np.sum((point2 - point1)**2))

def get_edges_h(c, p1, p2, vis=False):
    '''
    h = get_edges_h(np.array([0, 0]), np.array([-1, 10]), np.array([1,10.5]), True)
    '''
    dist_c = 'blue'
    pm = midpoint(p1, p2)
    p1_proj = projection_onto_line(p1, c, pm)
    p2_proj = projection_onto_line(p2, c, pm)
    edges_h = pts_distance(p1_proj, p2_proj)
    if p2_proj[1] > p1_proj[1]:
        edges_h = -edges_h
        dist_c = 'red'

    if vis:
        f, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(p1[0], p1[1], marker='*', color='green')
        ax.scatter(p2[0], p2[1], marker='*', color='blue')
        ax.scatter(c[0], c[1], color='black')  
        ax.scatter(pm[0], pm[1], marker='+', color='black')
        ax.scatter(p1_proj[0], p1_proj[1], marker='+', color='green')
        ax.scatter(p2_proj[0], p2_proj[1], marker='+', color='blue')
        ax.plot([c[0], pm[0]], [c[1], pm[1]], '--', color='black', linewidth=.3)
        ax.plot([p1_proj[0], p2_proj[0]], [p1_proj[1], p2_proj[1]], color=dist_c, linewidth=4, alpha=.2, zorder=0)
        ax.axis('equal')
        plt.title(f'{edges_h:.3f}')
        plt.show()
    return edges_h

def find_circle_center(x, y):
    x1, y1 = x[0], y[0]
    x2, y2 = x[1], y[1]
    x3, y3 = x[2], y[2]
    
    s1 = x1**2 + y1**2
    s2 = x2**2 + y2**2
    s3 = x3**2 + y3**2
    M11 = x1*y2 + x2*y3 + x3*y1 - (x2*y1 + x3*y2 + x1*y3)
    M12 = s1*y2 + s2*y3 + s3*y1 - (s2*y1 + s3*y2 + s1*y3)
    M13 = s1*x2 + s2*x3 + s3*x1 - (s2*x1 + s3*x2 + s1*x3)
    x0 =  0.5*M12/M11
    y0 = -0.5*M13/M11
    r0 = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
    return x0, y0, r0

def n_secuential_sample(array, nparts=None, ntimes=None):
    l = len(array)
    if nparts:
        part1 = l // nparts
        m1 = np.random.randint(0, nparts)
        indexes = [m1+part1*i for i in range(nparts)]
    if ntimes:
        m1 = np.random.randint(0, ntimes)
        indexes = [m1+ntimes*i for i in range(l // ntimes)]
    return array[indexes]

def circle_error(params, points):
    xc, yc, r = params
    distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r
    return np.sum(distances**2)

def fit_circle(points):
    # Adivinar el centro y el radio inicial
    x0 = np.mean(points[:, 0])
    y0 = np.mean(points[:, 1])
    r0 = np.mean(np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2))

    # Minimizar el error para obtener el mejor ajuste del círculo
    result = minimize(circle_error, [x0, y0, r0], args=(points,), method='Nelder-Mead')

    # Obtener los parámetros del círculo ajustado
    xc, yc, r = result.x

    return xc, yc, r

def filt_pts_in_ring(points, x_1, y_1, R_1, R_tol, vis= False):
    x = points[:, 0]
    y = points[:, 1]
    r_1_inner, r_1_outer = R_1-R_tol, R_1+R_tol
    dists = np.sqrt((x - x_1)**2 + (y - y_1)**2)
    pts_in_ring = points[np.logical_and(dists >= r_1_inner, dists <= r_1_outer)]

    if vis:
        f, ax = plt.subplots(figsize=(8,8))
        ax.scatter(x, y, s = .2)
        ax.scatter(pts_in_ring[:,0], pts_in_ring[:,1], s = 1)
        circulo = plt.Circle((x_1, y_1), R_1, color='black', fill=False, linewidth=4, alpha=.3)
        ax.add_artist(circulo)
        circulo = plt.Circle((x_1, y_1), r_1_inner, color='black', fill=False)
        ax.add_artist(circulo)
        circulo = plt.Circle((x_1, y_1), r_1_outer, color='black', fill=False)
        ax.add_artist(circulo)
        ax.axis('equal')
        plt.show()
    return pts_in_ring

def filt_pts_in_2_rings(points, x_1, y_1, R_1, x_2, y_2, R_2, R_tol, vis=False):
    x = points[:, 0]
    y = points[:, 1]

    dists_1 = np.sqrt((x - x_1)**2 + (y - y_1)**2)
    dists_2 = np.sqrt((x - x_2)**2 + (y - y_2)**2)

    r_1_inner, r_1_outer = R_1 - R_tol, R_1 + R_tol
    r_2_inner, r_2_outer = R_2 - R_tol, R_2 + R_tol

    points_in_1 = points[np.logical_and(dists_1 >= r_1_inner, dists_1 <= r_1_outer)]
    points_in_2 = points[np.logical_and(dists_2 >= r_2_inner, dists_2 <= r_2_outer)]

    return np.unique(np.concatenate([points_in_1, points_in_2]), axis=0)

def moving_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=data)
    return smoothed_data

def get_XYZ(file_name):


    with open(file_name) as f:
        file_content = f.readlines()
    file_content = [x.strip() for x in file_content]
    data_array = []
    for line in file_content:
        s = line.split(' ')
        if len(s) == 4 or len(s) == 3:
            data_array.append(s)

    data_array = np.array(data_array[1:])

    indices_ordenados = np.lexsort((data_array[:, 0], data_array[:, 1]))
    data_array = data_array[indices_ordenados]

    XYZ = np.zeros(shape = (20000,3000,3))
    y_anterior= data_array[0,1]
    perf = 0
    x_pos = 0
    for line in range (0,data_array.shape[0]):
        y_actual = data_array[line,1]
        if y_actual != y_anterior:
            perf = perf + 1
            x_pos = 0
        XYZ[perf,x_pos,:] = data_array[line,:-1]
        x_pos = x_pos + 1
        y_anterior = y_actual

    print(f'Numero de perfiles: {perf}')
    print(f'El shape de la array es de: {XYZ.shape}')
    return XYZ

def get_prof_xy_data(prof_n, XYZ):
        points = XYZ[prof_n,:,[0,2]].T
        return points[np.argsort(points[:, 0])]

def visualizar_puntos_en_orden(points):
    z = np.linspace(0, 1, len(points)) 
    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:,0], points[:,1], c=z, cmap='Reds')
    ax.axis('equal')
    plt.show()

def filter_coordinates(array_coords, x_range, y_range):
    x_mask  = np.logical_and(array_coords[:, 0] >= x_range[0], array_coords[:, 0] <= x_range[1])
    y_mask  = np.logical_and(array_coords[:, 1] >= y_range[0], array_coords[:, 1] <= y_range[1])
    total_mask = np.logical_and(x_mask, y_mask)
    return array_coords[total_mask]

def get_2_circ_r_and_xyc(points, points_filt, n_prof, perc_list = [0, .3, .7, 1], R_tol=.3, vis=False):
        data_len = len(points_filt)
        a = np.floor(data_len*perc_list[0]).astype(int)
        b = np.floor(data_len*perc_list[1]).astype(int)
        c = np.ceil(data_len*perc_list[2]).astype(int)
        d = np.ceil(data_len*perc_list[3]).astype(int)
        data_1 = points_filt[a:b]
        data_2 = points_filt[c:d]
        xc1, yc1, r1 = fit_circle(data_1)
        xc2, yc2, r2 = fit_circle(data_2)
        if (r1 < 10 or r1 > 22) or (r2 < 10 or r2 > 22):
            output = None
            vis = True
        else:
            ps_en_aro = filt_pts_in_2_rings(points, xc1, yc1, r1, xc2, yc2, r2, R_tol, vis=True)
            output = ps_en_aro, xc1, yc1, r1, xc2, yc2, r2
        if vis:
                f, ax = plt.subplots(figsize=(8,8))
                ax.scatter(points[:,0], points[:,1], s = .1, color='black')
                if output != None: ax.scatter(ps_en_aro[:,0], ps_en_aro[:,1], s = 5, color='green')

                circulo = plt.Circle((xc1, yc1), r1, color='blue', fill=False, linewidth=4, alpha=.3)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc1, yc1), r1+R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc1, yc1), r1-R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)

                circulo = plt.Circle((xc2, yc2), r2, color='green', fill=False, linewidth=4, alpha=.3)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc2, yc2), r2+R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc2, yc2), r2-R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)

                ax.axvspan(points_filt[a, 0], points_filt[b, 0], color='lightgrey', alpha=.5, zorder=0)
                ax.axvspan(points_filt[c, 0], points_filt[d-1, 0], color='lightgrey', alpha=.5, zorder=0)

                ax.set_title(f'Radio_1: {r1:.2f} | xy_coords_1: {xc1:.2f} / {yc1:.2f}\nRadio_2: {r2:.2f} | xy_coords_2: {xc2:.2f} / {yc2:.2f}')
                ax.axis('equal')
                if output == None:
                    plt.savefig(f"perfiles_malos/Prof_{n_prof+1}.png")
                    plt.close()
                plt.show()
        return output

def get_circ_r_and_xyc(points, points_filt, perc_list = [0, .3, .7, 1], R_tol=.3, vis=False):
        data_len = len(points_filt)
        a = np.floor(data_len*perc_list[0]).astype(int)
        b = np.floor(data_len*perc_list[1]).astype(int)
        c = np.ceil(data_len*perc_list[2]).astype(int)
        d = np.ceil(data_len*perc_list[3]).astype(int)
        data = np.concatenate((points_filt[a:b], points_filt[c:d]), axis=0)
        xc, yc, r = fit_circle(data)
        if r < 10 or r > 22: return None
        ps_en_aro = filt_pts_in_ring(points, xc, yc, r, R_tol, vis= False)
        if vis:
                f, ax = plt.subplots(figsize=(8,8))
                ax.scatter(points[:,0], points[:,1], s = .1, color='black')
                ax.scatter(ps_en_aro[:,0], ps_en_aro[:,1], s = 5, color='green')

                circulo = plt.Circle((xc, yc), r, color='blue', fill=False, linewidth=4, alpha=.3)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc, yc), r+R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)
                circulo = plt.Circle((xc, yc), r-R_tol, color='black', fill=False, linewidth=1)
                ax.add_artist(circulo)

                ax.axvspan(points_filt[a, 0], points_filt[b, 0],color='lightgrey', alpha=.5, zorder=0)
                ax.axvspan(points_filt[c, 0], points_filt[d-1, 0],color='lightgrey', alpha=.5, zorder=0)

                ax.set_title(f'Radio_1: {r:.2f} | xy_coords_1: {xc:.2f} / {yc:.2f}')
                ax.axis('equal')
                ax.axvline(xc)
                plt.show()
        return ps_en_aro, xc, yc, r

def vis_prop_w_mov_av(data, wdw, ylims=None):
    if isinstance(data, list): data = np.array(data)
    mov_av = np.convolve(np.array(data), np.ones(wdw)/wdw, mode='valid')
    range_x = np.arange(len(mov_av)) + (wdw//2)
    f, ax = plt.subplots(figsize=(8, 8))
    ax.plot(data, alpha=.2)
    ax.plot(range_x, mov_av, color='red')
    if ylims: ax.set_ylim(ylims)
    plt.show()

def get_gap_pts(points):
    points_x = points[:,0]
    diferencias = np.diff(points_x)
    indice_mayor_dif = np.argsort(np.abs(diferencias))[-1:]
    return points[[indice_mayor_dif[0], indice_mayor_dif[0]+1],:]

def visualizar(points, prof_info, xlims = None, ylims = None, savefig=False, folder='Imagenes'):
    f, ax = plt.subplots(figsize=(8,8))
    ax.scatter(points[:,0], points[:,1], s = 1)
    ax.scatter(prof_info['gap_points'][:,0], prof_info['gap_points'][:,1], s = 100, color='blue', marker='o', facecolors='none')
    ax.plot(prof_info['gap_points'][:,0], prof_info['gap_points'][:,1], linestyle='--', color='black')
    if xlims: ax.set_xlim(xlims)
    if ylims: ax.set_xlim(ylims)
    ax.axis('equal')
    plt.title(f"Prof: {prof_info['n_prof']} | w: {prof_info['w']:.2f} | w: {prof_info['h']:.2f} ")
    if savefig:
        plt.savefig(f"{folder}/Prof_{prof_info['n_prof']}.png")
        plt.close()
    else:
        plt.show()

def remove_outliers_IQR_meth(data):
    if isinstance(data, list): data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    l_bound = Q1 - 1.5 * IQR
    u_bound = Q3 + 1.5 * IQR
    
    return data[(data >= l_bound) & (data <= u_bound)]


def get_gap_deltaR_David_1(data, n_profiles, f_info, xc_mean, yc_mean, r_mean, savefig = False):
    gaps = []
    deltaRs = []
    for prof in range(n_profiles):
        prof_info = {}
        prof_info['n_prof'] = prof+1
        try:
            orig_points = get_prof_xy_data(prof, data)
            points_filt_1 = filter_coordinates(orig_points, *f_info)
            points_filt_1 = filt_pts_in_ring(points_filt_1, xc_mean, yc_mean, r_mean, 1, vis= False)
            data_mov_av = moving_average(points_filt_1, 5)
            r_and_xyc = get_2_circ_r_and_xyc(points_filt_1, data_mov_av, prof, perc_list = [0, .35, .65, 1], R_tol=.3, vis=False)
            if r_and_xyc is None:
                print(f'Perfil {prof} fuera de los limites')
            else:
                ps_en_aro, xc1, yc1, r1, xc2, yc2, r2 = r_and_xyc
                prof_info['gap_points'] = get_gap_pts(ps_en_aro)
                w = pts_distance(prof_info['gap_points'][0], prof_info['gap_points'][1])
                prof_info['w'] = w
                h = get_edges_h(np.array([np.array([xc1, xc2]).mean(), np.array([yc1, yc2]).mean()]), prof_info['gap_points'][0], prof_info['gap_points'][1], vis=False)
                prof_info['h'] = h
                if savefig: visualizar(points_filt_1, prof_info, xlims = [-5,5], ylims = [110, 120], savefig=True, folder='Imagenes_2')
                gaps.append(w)
                deltaRs.append(h)
        except Exception as e:
                print(f"Error en el perfil: {prof} | Error: {type(e).__name__} : {e}")
    return gaps, deltaRs

def get_xc_yc_r_mean(XYZ, n_profiles, filt_info, vis=False):
    xcs = []
    ycs = []
    rs = []
    for prof in range(n_profiles):
        try:
                orig_points = get_prof_xy_data(prof, XYZ)
                points_filt_1 = filter_coordinates(orig_points, *filt_info)
                data_mov_av = moving_average(points_filt_1, 10)
                r_and_xyc2 = get_circ_r_and_xyc(points_filt_1, data_mov_av, perc_list = [0, .3, .7, 1], R_tol=.3, vis=False)
                if r_and_xyc2 is None:
                        print(f'Perfil {prof} fuera de los limites')
                else:
                        ps_en_aro, xc, yc, r = r_and_xyc2
                        xcs.append(xc)
                        ycs.append(yc)
                        rs.append(r)
        except Exception as e:
                print(f"Error en el perfil: {prof} | Error: {type(e).__name__} : {e}")
    xcs_out_r = remove_outliers_IQR_meth(xcs)
    ycs_out_r = remove_outliers_IQR_meth(ycs)
    rs_out_r = remove_outliers_IQR_meth(rs)
    xc_mean = xcs_out_r.mean()
    yc_mean = ycs_out_r.mean()
    r_mean = rs_out_r.mean()
    if vis:
        f,ax = plt.subplots(1, 3, figsize=(12,5))
        ax[0].plot(xcs)
        ax[0].plot(xcs_out_r)
        ax[0].axhline(xc_mean, color='black')

        ax[1].plot(ycs)
        ax[1].plot(ycs_out_r)
        ax[1].axhline(yc_mean, color='black')

        ax[2].plot(rs)
        ax[2].plot(rs_out_r)
        ax[2].axhline(r_mean, color='black')

        ax[0].set_title(f'xcs: {xc_mean:.2f}')
        ax[1].set_title(f'ycs: {yc_mean:.2f}')
        ax[2].set_title(f'rs: {r_mean:.2f}')
        plt.show()
    return xc_mean, yc_mean, r_mean
