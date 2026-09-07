# from __future__ import annotations

# bad_srs = {
#     400213: [0],
#     400215: [52, 54, 55]
# }

from pathlib import Path
import argparse
# import shutil
# import tempfile

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import matplotlib. patheffects as fx

import time as tm

import pandas as pd
import yaml
import sep
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import os
import target_io
from numba import njit
import glob
from datetime import datetime, timezone
import re

# from PIL import Image
import astroalign as aa


from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import csv


make_nans = False

pxs_p_quad = 16 # Part of the camera geometry builder: pixels per quadrants
quads_p_module = 4 # Part of the camera geometry builder: quadrants per module

sources_of_interest = [  # List your sources of interest, like Mrk421
    {'name': 'Mrk421', 'ra': 166.11380833, 'dec': 38.208833, 'mag': 12.9},
    # {'name': '51 UMa', 'ra': 166.13016, 'dec': 38.241365, 'mag': 6.28},
    ]

others_bounds = [[0, np.exp(4)], [0, np.exp(4)]] # "Others" event box bounds, to select dim events for the star images

fpms_t = ['7-21', '7-22', '7-23', '7-24', '7-15', '7-16', '7-17', '7-18', '7-19', '7-10', '7-11', '7-12', '7-13', '7-14', '7-5', '7-6', '7-7', '7-8', '7-9', '7-0', '7-1', '7-20', '7-2', '7-3', '7-4'] # Order of the FPMs in the data files


ups_mods = { # Note this list modules that do not exist. This is because I just care about identifying positions which would flip upsidown if they were to be placed. 
    "0": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "2": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "3": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "5": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "6": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "8": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "1": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
    "4": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
    "7": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
}

fpms_exist = { # Which modules actually exist in each sector
    "0": [8, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    "2": [5, 6, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23],
    "3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "6": [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 18, 19],
    "8": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 16],
    "1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
}

def load_csv(path):
    result = pd.read_csv(path)
    return result

def load_config(filepath): # Function to load a config file. Should change to yaml at some point
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
    return result

@njit
def getarrs(wfs_a): # Gives two arrays: one of the value of the mean charge per pixel per event, the other is the mean charge^2 per pixel per event
    wfs_me = [np.zeros((1600), dtype=np.float64) for _ in wfs_a]
    wfs_sq = [np.zeros((1600), dtype=np.float64) for _ in wfs_a]
    for ev, wf in enumerate(wfs_a):
        for ch in range(wf.shape[0]):
            wfs_me[ev][ch] = np.mean(wf[ch])
            wfs_sq[ev][ch] = np.mean(wf[ch]**2)
    return wfs_me, wfs_sq

@njit
def get_gbmean(wfs_m, wfs_sm): # Gets the mean of a matrix of some shape, over the first dimension
    N = len(wfs_m)
    gbmean = np.zeros(wfs_m[0].shape)
    for wf in wfs_m:
        gbmean = gbmean + wf
    return gbmean/N

@njit
def get_int_chargemeanstd_ev(wf, int_win=2): # Gets mean and std charge of all pixels per event
    n_channels, n_samples = wf.shape
    int_charge = np.zeros((n_channels))
    ##### HARDCODED MASKING
    wf[4*64 + 14, :] = np.nan
    wf[5*64:6*64, :] = np.nan
    wf[13*64:14*64, :] = np.nan
    wf[21*64:, :] = np.nan
    ##### ^^^^^^
    for ch in range(n_channels):
        t_max = np.argmax(wf[ch])
        if t_max > n_samples - int_win - 1:
            t_max = n_samples - int_win - 1
        elif t_max < int_win:
            t_max = int_win
        int_charge[ch] = wf[ch, t_max-int_win:t_max+int_win+1].sum()
    mean_c = np.nanmean(int_charge)
    std_c = np.nanstd(int_charge)
    
    return mean_c, std_c

def read_wfs_metrics(calfile, save=False, reader = None, filter = True): # Calculate the waveforms means and mean of the squares, per pixel per event (this filters events with strong charges out)
    if reader == None:
        reader = target_io.WaveformArrayReader(calfile, silent=True)
    wfs_me = []
    wfs_sq = []
    times = []
    wfs_all = []
    for ev in range(reader.fNEvents):
        wfs = np.zeros((reader.fNPixels, reader.fNSamples), dtype=np.float32)

        reader.GetR1Event(ev, wfs)
        if filter:
            mean_c, std_c = get_int_chargemeanstd_ev(wfs, 4)
            if mean_c < others_bounds[0][1] and std_c < others_bounds[1][1]:
                wfs_all.append(wfs)
                times.append(reader.fTACK_time)
        else:
            wfs_all.append(wfs)
            times.append(reader.fTACK_time)
    aa, bb = getarrs(wfs_all)
    wfs_me = wfs_me + aa
    wfs_sq = wfs_sq + bb
        
    # all_wfs = np.array(all_wfs)
    
    for wf_m, wf_s in zip(wfs_me, wfs_sq):
        wf_m[4*64 + 14] = 0.0
        # for l in range(64):
        wf_m[5*64:6*64] = 0.0
        wf_m[13*64:14*64] = 0.0
        wf_m[21*64:] = 0.0

        wf_s[4*64 + 14] = 0.0
        # for l in range(64):
        wf_s[5*64:6*64] = 0.0
        wf_s[13*64:14*64] = 0.0
        wf_s[21*64:] = 0.0
    return np.array(wfs_me), np.array(wfs_sq), np.array(times)

def length_to_pixel(c, module_length = 52.05, module_pitch = 54): # Converts length coordinate to pixel coordinate

    interval = (module_pitch - module_length)
    pix = ((c - interval*(((c//(module_pitch/2))+1)//2))/module_length)*8
    return pix

def pixel_to_length(pix, module_length = 52.05, module_pitch = 54): # Converts pixel coordinate to length coordinate
    

    interval = (module_pitch - module_length)
    c = (pix)*(module_length)/8 + interval*((((pix)//4)+1)//2)
    return c

def CTC_lab_im_map(): # Function used to map the 64-long array of a module to the 8x8 grid of a physical module
    """
    Calculates the grid index, which is used to go from (64) pixel data to (8, 8) lab image. Includes TARGET to SiPM mapping.

    :return: grid index
    :rtype: numba.typed.List
    """

    ch_nums = np.array([[23,22,19,18, 4, 5, 0, 1],
                        [21,20,17,16, 6, 7, 2, 3],
                        [30,31,27,26,12,13, 8, 9],
                        [29,28,25,24,14,15,10,11],
                        [55,54,51,50,36,37,32,33],
                        [53,52,49,48,38,39,34,35],
                        [62,63,59,58,44,45,40,41],
                        [61,60,57,56,46,47,42,43]])
    ch_nums_1D = ch_nums.reshape(-1)
    ch_to_pos = dict(zip(ch_nums_1D, np.arange(64)))

    total_cells = 64
    indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
    grid_ind = list()

    i, j = 0, 0
    ch_map = dict()
    ch_map = ch_to_pos
    pix_ind = np.array(indices[(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
    for asic in range(4):
        for ch in range(16):
            grid_ind.append(int(pix_ind[ch_map[asic * 16 + ch]]))
            
    return grid_ind

grid_ind0 = CTC_lab_im_map() # Reference mapping matrix for going from the 1x64 to 8x8

def make_grid_module(vals, a = None, x0 = 0, y0 = 0, upsideup = True, grid_ind = grid_ind0): # Makes the grid of a module
# grid_ind = CTC_lab_im_map()
    try:
        bb = a[0][0]
    except:
        a = np.empty(shape = (8+x0, 8+y0))
        if make_nans: a[:] = np.nan
        else: a[:] = 0.0
    bb = a.shape
    if bb[0] < x0 + 8:
        b = a
        a = np.empty(shape = (x0+8, a.shape[1]))
        a[:b.shape[0]][:b.shape[1]] = b
        bb = a.shape
    if bb[1] < y0 + 8:
        b = a
        a = np.empty(shape = (a.shape[0], y0+8))
        a[:b.shape[0]][:b.shape[1]] = b

    # vals = np.array(vals)
    # if len(vals) < 64:
    #     vals = np.pad(vals.astype(float), (0, 64 - len(vals)), constant_values=np.nan)
    if all(x == 0.0 for x in vals) and make_nans:
        vals[:] = np.nan
    for i, val in enumerate(vals):
        xy = grid_ind[i]
        y, x = xy // 8, xy % 8
        # x, y = xy % 8, xy // 8
        if upsideup:
            y =  7 - y
            x = 7 - x
        a[x0+x, y0+y] = val
    return np.array(a)
    # return np.array(list(reversed(list(list(zip(*a))))))

def make_grid_sector(valss, fpms,  a = None, x0 = 0, y0 = 0, sector = None, grid_ind = grid_ind0): # Makes the grid of a sector
    try:
        bb = a[0][0]
    except:
        a = np.empty(shape = (40+x0, 40+y0))
        a[:] = np.nan
    bb = a.shape
    if bb[0] < x0 + 40:
        b = a
        a = np.empty(shape = (x0+40, a.shape[1]))
        a[:b.shape[0]][:b.shape[1]] = b
    bb = a.shape
    if bb[1] < y0 + 40:
        b = a
        a = np.empty(shape = (a.shape[0], y0+40))
        a[:b.shape[0]][:b.shape[1]] = b
    valss = [vals for vals in valss]
    if not make_nans:
        if sector == None and len(fpms) > 0:
            sector, _ = fpms[0].split("-")
        elif sector == None:
            sector = "4" # this will be a blank picture I think
        for fp in fpms_exist[sector]:
            if f"{sector}-{fp}" not in fpms:
                fpms.append(f"{sector}-{fp}")
                valss.append([0.0 for _ in range(pxs_p_quad*quads_p_module)])

    for i, vals in zip(fpms, valss):
        sector, i = i.split("-")
        i = int(i)
        x00 = (i*8 % 40)
        y00 = (8*(4 - (i*8 // 40)))
        if i in ups_mods[str(sector)]:
            upsideup = True
        else:
            upsideup = False
        a = make_grid_module(vals, a, x0+x00, y0+y00, upsideup, grid_ind = grid_ind)
    return np.array(a)

def make_grid_camera(valss, fpms,  a = None, grid_ind = grid_ind0): # Makes the grid of the full camera
    a = np.empty(shape = (120, 120))
    a[:] = np.nan

    fpmss = {"0": [],
             "1": [],
             "2": [],
             "3": [],
             "4": [],
             "5": [],
             "6": [],
             "7": [],
             "8": [],
    }

    valsss = {"0": [],
             "1": [],
             "2": [],
             "3": [],
             "4": [],
             "5": [],
             "6": [],
             "7": [],
             "8": [],
    }

    for val, fpm in zip(valss, fpms[:len(valss)]):
        sector, _ = fpm.split("-")
        fpmss[sector].append(fpm)
        valsss[sector].append(val)

    for i in range(9):
        sector = i
        i = f"{i}"
        fpms = fpmss[i]
        valss = valsss[i]
        # sector, _ = fpms[0].split("-")
        # sector = int(sector)
        x0 = (sector*40) % 120
        y0 = 80 - 40*(sector // 3)
        a = make_grid_sector(valss, fpms, a, x0, y0, grid_ind = grid_ind, sector = i)
    # return np.array(a)
    return np.array(list(reversed(list(list(zip(*a))))))

class CameraDisplay: # The class that is the camera display for the pSCT
    """
    similar to ctapipe, creating a instance of camera frame as a class so that it is easier to update.
 
    Parameters
    ----------

    values : array-like, optional
        Initial scalar value per patch. If not provided, defaults to
        zeros in the same order as `patches`.
    cmap : str or Colormap, optional
        Colormap used to convert values -> colors. Default 'viridis'.
    vmin, vmax : float, optional
        Fixed color-scale limits. If None, autoscaled from `values`
        each time `set_values` is called (unless `autoscale=False`).
    **collection_kwargs :
        Extra kwargs passed to PatchCollection (e.g. edgecolor, lw, alpha).
    """
 
    def __init__(self, frame=None, cmap="viridis",  # 
                 vmin=None, vmax=None, autoscale=True, mod_length = 52.05, mod_pitch = 54.00, pix_size = 6.50625, **collection_kwargs):
        base_frame = make_grid_camera([[]], fpms=[])

        grid_size = (mod_length/8)
        artists = []
        mapping = []
        space = (grid_size - pix_size)/2
        for l in range(base_frame.shape[0]):
            for k in range(base_frame.shape[1]):
                if base_frame[k, l] != np.nan:
                    artists.append(mpatches.Rectangle((pixel_to_length(l-60, mod_length, mod_pitch)+space, pixel_to_length(k-60, mod_length, mod_pitch)+space), pix_size, pix_size, ec="none"))
                    mapping.append([l, k])
        self.baseframe = base_frame
        self.patches = artists
        self.n = len(self.patches)
        self.mapping = mapping
 
        if frame is None:
            values = np.zeros(self.n)
        else:
            values = []
            for l in range(base_frame.shape[0]):
                for k in range(base_frame.shape[1]):
                    if base_frame[k, l] != np.nan:
                        values.append(frame[k, l])
        values = np.asarray(values, dtype=float)
        if values.shape[0] != self.n:
            raise ValueError(
                f"len(values)={values.shape[0]} does not match "
                f"len(patches)={self.n}"
            )
 
        self.autoscale = autoscale and (vmin is None or vmax is None)
        self.norm = Normalize(vmin=vmin, vmax=vmax)
 
        self.collection = PatchCollection(
            self.patches, cmap=cmap, **collection_kwargs
        )
        self.collection.set_norm(self.norm)
        self.collection.set_array(values)
 
        self._ax = None
        self._cbar = None
 
    # ------------------------------------------------------------------
    # Value / color updates
    # ------------------------------------------------------------------
    def set_values(self, frame):
        """Update all patch values at once (vectorized, fast)."""
        values = np.asarray([frame[obj[1], obj[0]] for obj in self.mapping], dtype=float)
        if values.shape[0] != self.n:
            raise ValueError(
                f"len(values)={values.shape[0]} does not match "
                f"len(patches)={self.n}"
            )
        self.collection.set_array(values)
 
        if self.autoscale:
            self.collection.autoscale()
 
        self._redraw()
 
    def set_value(self, index, value):
        """Update a single patch's value by index."""
        arr = self.collection.get_array().copy()
        arr[index] = value
        self.set_values(arr)
 
    def get_values(self):
        return np.asarray(self.collection.get_array())
 
    def set_cmap(self, cmap):
        self.collection.set_cmap(cmap)
        self._redraw()
 
    def set_clim(self, vmin, vmax):
        """Fix the color scale limits (disables autoscaling)."""
        self.autoscale = False
        self.collection.set_clim(vmin, vmax)
        self._redraw()
 
    def _redraw(self):
        if self._ax is not None:
            self._ax.figure.canvas.draw_idle()
 
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, ax=None, colorbar=True, cbar_label=None,
              autoscale_view=True, **cbar_kwargs):
        """
        Plot the geometry on `ax` (creating a new figure/axes if None).
        Safe to call once; afterwards use `set_values` to update in place.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = (6,6))
        else:
            fig = ax.figure
            
        ax.add_collection(self.collection)
 
        if autoscale_view:
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
 
        if colorbar:
            self._cbar = fig.colorbar(self.collection, ax=ax, **cbar_kwargs)
            if cbar_label:
                self._cbar.set_label(cbar_label)
        fig.set_dpi(500)
        ax.set_xlim(-420, 420)
        ax.set_ylim(-420, 420)
        ax.set_xlabel('Horizontal [mm]')
        ax.set_ylabel('Vertical [mm]')
        ax.set_aspect('equal')
        self._ax = ax
        self._fig = fig
        return fig, ax
    
    # ------------------------------------------------------------------
    # Animation / saving
    # ------------------------------------------------------------------
    def animate(self, values_list, ax=None, interval=200, colorbar=True,
                cbar_label=None, global_min = None, global_max = None, repeat=True, blit=False, plot_title = None, **anim_kwargs):
        """
        Build an animation that steps through a list of value-arrays,
        one per frame, calling `set_values` on each frame.
 
        Parameters
        ----------
        values_list : list/array of array-like
            Sequence of value arrays, one per animation frame. Each
            entry must have length == number of patches.
        ax : matplotlib Axes, optional
            Reused if the geometry hasn't been plotted yet (falls back
            to `plot()`'s defaults). Ignored if already plotted.
        interval : int
            Delay between frames in milliseconds.
        colorbar, cbar_label :
            Passed to `plot()` if the geometry hasn't been plotted yet.
        repeat, blit, **anim_kwargs :
            Passed through to matplotlib.animation.FuncAnimation.
 
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Also stored on self._anim (used by `save()`).
        """
        values_list = [np.asarray(v, dtype=float) for v in values_list]
        if global_min == None:
            global_min = np.nanmin(values_list)
        if global_max == None:
            global_max = np.nanmax(values_list)
 
        if self._ax is None:
            self.plot(ax=ax, colorbar=colorbar, cbar_label=cbar_label)
 
        fig = self._ax.figure

        self.set_clim(global_min, global_max)
 
        def _update(frame_values):
            self.set_values(frame_values)
            return (self.collection,)
 
        self._anim = FuncAnimation(
            fig,
            _update,
            frames=values_list,
            interval=interval,
            blit=blit,
            repeat=repeat,
            **anim_kwargs,
        )
        fig.suptitle(plot_title)
        return self._anim
 
    def save(self, filename, fps=10, dpi=150, writer=None, **kwargs):
        """
        Save the current state.
 
        - If `animate()` was called and an animation exists, saves the
          animation (e.g. .gif via 'pillow', .mp4 via 'ffmpeg').
        - Otherwise saves whatever is currently plotted as a static image.
        """
        anim = getattr(self, "_anim", None)
 
        if anim is not None:
            if writer is None:
                writer = "pillow" if str(filename).lower().endswith(".gif") else "ffmpeg"
            anim.save(filename, writer=writer, fps=fps, dpi=dpi, **kwargs)
        else:
            if self._ax is None:
                raise RuntimeError(
                    "Nothing to save: call plot() or animate() first."
                )
            self._ax.figure.savefig(filename, dpi=dpi, **kwargs)

def physical_image(frame, mod_length = 52.05, pix_size = 6.50625, colorscheme = 'viridis', fig = None, ax = None): # Old function that made the physical image of the camera. Outdated now with the class CameraDisplay

    grid_size = (mod_length/8)
    space = (grid_size - pix_size)/2
    max_val = np.nanmax(frame)
    min_val = np.nanmin(frame)
    # mpl.colormaps["viridis"]((v-min_val)/(max_val-min_val))
    colors = []
    vals = []
    artists_ind = []
    for l in range(frame.shape[0]):
        for k in range(frame.shape[1]):
            if frame[k, l] != np.nan:
                col = mpl.colormaps[colorscheme]((frame[k, l]-min_val)/(max_val-min_val))
                colors.append(col)
                vals.append(frame[k, l])
                # artists[l, k] = mpatches.Rectangle((pixel_to_length(l-60)+space, pixel_to_length(k-60)+space), pix_size, pix_size, ec="none", color = col)
                artists_ind.append([l, k])
                

    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize = (6,6))
        for col, obj in zip(colors, artists_ind):
            l, k = obj
            ax.add_artist(mpatches.Rectangle((pixel_to_length(l-60)+space, pixel_to_length(k-60)+space), pix_size, pix_size, ec="none", color = col))
        ax.set_xlim(-450, 450)
        ax.set_ylim(-450, 450)
        ax.set_xlabel('Horizontal [mm]')
        ax.set_ylabel('Vertical [mm]')
        ax.set_aspect('equal')
        fig.set_dpi(500)
    else: # This block is supposed to just update the colors of the pixels in a already formed plot, in attempts of making an image faster, but it is not working so don't use it
        ind = 0
        for col, obj in zip(colors, artists_ind):
            ax.get_children()[ind].set_facecolor(col)
            ind += 1
    return fig, ax

def physical_image_class(frame, mod_length = 52.05, pix_size = 6.50625, colorscheme = 'viridis', fig = None, ax = None): # Outdated function trying to use the class

    grid_size = (mod_length/8)
    space = (grid_size - pix_size)/2
    max_val = np.nanmax(frame)
    min_val = np.nanmin(frame)
    # mpl.colormaps["viridis"]((v-min_val)/(max_val-min_val))
    colors = []
    artists = []
    vals = []
    artists_ind = []
    for l in range(frame.shape[0]):
        for k in range(frame.shape[1]):
            if frame[k, l] != np.nan:
                col = mpl.colormaps[colorscheme]((frame[k, l]-min_val)/(max_val-min_val))
                colors.append(col)
                vals.append(frame[k, l])
                # artists[l, k] = mpatches.Rectangle((pixel_to_length(l-60)+space, pixel_to_length(k-60)+space), pix_size, pix_size, ec="none", color = col)
                artists_ind.append([l, k])
                artists.append(mpatches.Rectangle((pixel_to_length(l-60)+space, pixel_to_length(k-60)+space), pix_size, pix_size, ec="none"))

    camdisp = CameraDisplay(artists, values=vals, cmap="viridis",
                 vmin=None, vmax=None, autoscale=True)
    
    camdisp.plot()

    # return fig, ax

def rotate(a, nturns = 0, clockwise = True): # Rotates a matrix n 90 degree turns clockwise or counter clockwise
    if nturns > 0:
        for _ in range(nturns):
            if clockwise:
                a = list(reversed(list(list(zip(*a)))))
            else:
                a = list(reversed(list(list(zip(*a))[::-1])))
    return np.array(a)

def rotate_pair(turp, nturns = 0, clockwise = True): # Rotate a pair of points (x, y)
    for _ in range(nturns):
        if clockwise:
            turp = (turp[1], -turp[0])
        else:
            turp = (-turp[1], turp[0])
    return turp

def mirror_turp(turp, mirror = 'x'): # mirror a pair of points (x,y): 'x': (x, y) -> (-x, y); 'y': (x, y) -> (x, -y)
    if mirror == 'x':
        turp = (-turp[0], turp[1])
    if mirror == 'y':
        turp = (turp[0], -turp[1])
    return turp

def transforming(turp):
    return rotate_pair((-turp[0], turp[1]), nturns=1)

def get_visible_stars(ra_center, dec_center, fov_deg, xlen, ylen,  # Function that collects the stars from the catalog that are visible within a plate mapping defined by a field of view and a x-length and y-length, for a Alt/Az telescope
                       time_utc, lat, lon, elevation=0.0, catalog=None): 
    """
    Returns stars within a field of view as seen through an Alt/Az-mounted
    telescope: x increases toward azimuth-right (observer's right when
    facing the target), y increases toward zenith. Origin (0,0) is
    the center of the image. Field rotation over time falls out naturally—no separate
    rotation step needed.

    Written with help of Claude
    """
    if catalog is None:
        raise Exception(f"Provide a star catalog!")

    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elevation * u.m)
    obstime = Time(time_utc)                       # fresh each call
    altaz_frame = AltAz(obstime=obstime, location=location)  # fresh each call

    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame="icrs")
    names = [s["name"] for s in catalog]
    magnitudes = [s["mag"] for s in catalog]
    coords = SkyCoord(ra=[s["ra"] for s in catalog] * u.deg,
                       dec=[s["dec"] for s in catalog] * u.deg,
                       frame="icrs")

    half_fov = fov_deg / 2.0
    mask = coords.separation(center).deg <= half_fov
    if not np.any(mask):
        return []
    names = [n for n, m in zip(names, mask) if m]
    coords = coords[mask]

    # Transform to Alt/Az fresh, using this call's obstime/location
    altaz = coords.transform_to(altaz_frame)
    center_altaz = center.transform_to(altaz_frame)

    alt0, az0 = center_altaz.alt.rad, center_altaz.az.rad
    alt, az = altaz.alt.rad, altaz.az.rad

    cos_c = np.cos(alt0) * np.cos(alt) * np.cos(az - az0) + np.sin(alt0) * np.sin(alt)
    angg = np.arccos((np.sin(alt)-np.sin(alt0)*cos_c)/(np.cos(alt0)*(np.sin(np.arccos(cos_c)))))

    xi = np.cos(alt) * np.sin(az - az0) / cos_c
    eta = (np.cos(alt0) * np.sin(alt) - np.sin(alt0) * np.cos(alt) * np.cos(az - az0)) / cos_c
    xi_deg, eta_deg = np.rad2deg(xi), np.rad2deg(eta)


    px = (xi_deg + half_fov) / fov_deg * xlen
    py = (eta_deg + half_fov) / fov_deg * ylen   # origin bottom-left, y up toward zenith

    return [
        {
            "name": names[i],
            "ra": coords.ra.deg[i],
            "dec": coords.dec.deg[i],
            "alt": altaz.alt.deg[i],
            "az": altaz.az.deg[i],
            "x": float(px[i] - xlen/2),
            "y": float(py[i] - ylen/2),
            "mag": float(magnitudes[i])
        }
        for i in range(len(names))
    ]

def load_hyg_catalog(path, mag_limit=6.0, named_only=False): # Function that handles fetching the catalog objects from a .csv formatted properly
    """
    Loads stars from a HYG database CSV (e.g. hygdata_v3.csv).
    Download from: https://github.com/astronexus/HYG-Database

    path        : str   - path to hygdata_v3.csv
    mag_limit   : float - only include stars brighter than this apparent magnitude
    named_only  : bool  - if True, skip stars without a proper name
                          (most of the ~119,000 rows only have catalog IDs)

    Returns list of {"name", "ra", "dec", "mag"} dicts, ra/dec in degrees.

    Written with help of Claude
    """
    catalog = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mag = float(row["mag"])
            except (ValueError, KeyError):
                continue
            if mag > mag_limit:
                continue

            proper = (row.get("proper") or "").strip()
            if named_only and not proper:
                # print("A")
                continue

            try:
                ra_deg = float(row["ra"]) * 15.0   # HYG stores RA in hours
                dec_deg = float(row["dec"])
            except (ValueError, KeyError):
                continue

            if proper:
                # print("B")
                name = proper
            else:
                # fall back to Bayer/Flamsteed designation, then HD/HIP, then row id
                bayer = (row.get("bf") or "").strip()
                hd = (row.get("hd") or "").strip()
                hip = (row.get("hip") or "").strip()
                name = bayer or (f"HD {hd}" if hd else None) or (f"HIP {hip}" if hip else f"HYG {row.get('id')}")

            catalog.append({"name": name, "ra": ra_deg, "dec": dec_deg, "mag": mag})

    return catalog

def clean_image(a0, p1 = 8, p2 = 4): # This cleans a image from background so that star detection is easier
    
    return a0

def extract_timestamp(file_path, preceding_string, add_value = 0.0, wildcard="*"):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split on the wildcard, escape each part, rejoin with a regex wildcard
    parts = preceding_string.split(wildcard)
    pattern = r'.*?'.join(re.escape(part) for part in parts)
    pattern += r'([\d]+\.[\d]+)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return float(match.group(1)) + add_value
    else:
        raise ValueError(f"No number found after '{preceding_string}'")

def get_time_base(run_num, log_files): # Extracts the start time of a run from log file
    try:
        base_timestamp = extract_timestamp(log_files.format(run_num), "- INFO     - Clock origin timestamp: ")
    except:
        try:
            base_timestamp = extract_timestamp(log_files.format(run_num), "Sub-Run 0 Start Time: * / ")
        except:
            base_timestamp = None
    return base_timestamp

def convert_to_utc_str(time): # Converts the time to UTC time in string form
    utc_dt = datetime.fromtimestamp(time, tz=timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S")

def sexagesimal_to_degrees(coord_str, is_ra=False):
    """
    Convert a sexagesimal coordinate string to decimal degrees.

    Parameters
    ----------
    coord_str : str
        Coordinate string.
        - RA format:  "HH MM SS.SS"  (hours, minutes, seconds)
        - Dec format: "DD MM SS.SS"  (degrees, minutes, seconds; may have a
                       leading +/- sign, e.g. "-45 30 12.3")
    is_ra : bool
        Set True if coord_str is Right Ascension (hours), False if it's
        Declination (degrees). Default False (Dec).

    Returns
    -------
    float
        The coordinate in decimal degrees.
    """
    parts = coord_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Expected 'HH MM SS.SS' or 'DD MM SS.SS', got: {coord_str!r}")

    first_str, minutes_str, seconds_str = parts

    # Handle sign (relevant mainly for Dec, but harmless for RA)
    sign = -1.0 if first_str.strip().startswith('-') else 1.0
    first = abs(float(first_str))
    minutes = float(minutes_str)
    seconds = float(seconds_str)

    value = first + minutes / 60.0 + seconds / 3600.0

    if is_ra:
        value *= 15.0  # convert hours to degrees

    return sign * value



def fetch_pointing(run, sr = None, time = None, run_info_path = './pSCT_Run_Log-DR8_2026.csv'): # Will fetch the pointing of a run from the appropriate log files
    """
    Figure a function that takes the pointing 
    """
    run_info = load_csv(run_info_path)
    
    ra = run_info.loc[run_info['Run Number'] == run, 'Goal RA'].iloc[0]
    dec = run_info.loc[run_info['Run Number'] == run, 'Goal Dec'].iloc[0]
    return sexagesimal_to_degrees(ra, is_ra = True), sexagesimal_to_degrees(dec, is_ra = False) # ra, dec in degrees

def get_value_from_filename(filename, a, b):
    return int(filename.split(a, 1)[1].split(b, 1)[0])

def get_star_parameters_from_subrun_physical(a0, time_str, ra_center, dec_center, psct_config, catalog, sources = None, thresh = 1.5, minarea=5, filter_type='matched', deblend_nthresh=32, deblend_cont=0.005, clean=True, clean_param=1.0, segmentation_map=False,  distance_of_margin = 53, min_matches_fraction = 0.1, pixel_tol = 13, max_control_points = 50, detection_sigma = 5, min_area = 5): # This function acquires the many parameters that describes the star field and source field in the camera at a given subrun
    module_pitch = float(psct_config['module_pitch'])
    module_width = float(psct_config['module_width'])
    if sources != None:
        sources_list = sources
    else:
        sources_list = []
    a = clean_image(a0) # This cleans the 2D image. Need to improve
    a = a[:, ::-1] # Mirrors it horizontally (from camera view to sky view)
    a = rotate(a, 2, True) # Rotates 180 degrees - now the image matches the sky, since a 180 rotation is the effect of the two mirrors

    # Extracting objects in the image:
    a_pro = a.copy()
    a_pro[a_pro == 0.0] = np.nan # Pixels that did not fetch data are usually at 0.0 -> go to nan
    a_mask = a_pro.copy() # For a boolean mask
    a_mask[a_pro == np.nan] = False
    a_mask[a_pro != np.nan] = True
    # bkg = sep.Background(a, mask = a_mask) 
    objects = sep.extract(a, thresh = thresh, minarea=minarea, filter_type=filter_type, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, clean=clean, clean_param=clean_param, segmentation_map=segmentation_map, filter_kernel=None)
    # x_detected = np.array([pixel_to_length(obj[7]-60+0.5, module_pitch = module_pitch, module_length=module_width) for obj in objects])
    # y_detected = np.array([pixel_to_length(obj[8]-60+0.5, module_pitch = module_pitch, module_length=module_width) for obj in objects])
    x_detected = np.array([pixel_to_length(obj[7]-60, module_pitch = module_pitch, module_length=module_width) for obj in objects])
    y_detected = np.array([pixel_to_length(obj[8]-60, module_pitch = module_pitch, module_length=module_width) for obj in objects])

    # Listing possible stars in the field
    stars = get_visible_stars(
            ra_center=ra_center, dec_center=dec_center, fov_deg=float(psct_config['fov_deg']), # Function
            xlen=(14*module_pitch + module_width), ylen=(14*module_pitch + module_width),
            time_utc=time_str,
            lat=float(psct_config['lat']), lon=float(psct_config['lon']), elevation=float(psct_config['elevation']),
            catalog=catalog
        )
    
    sources_pred = get_visible_stars(
            ra_center=ra_center, dec_center=dec_center, fov_deg=float(psct_config['fov_deg']), # Function
            xlen=(14*module_pitch + module_width), ylen=(14*module_pitch + module_width),
            time_utc=time_str,
            lat=float(psct_config['lat']), lon=float(psct_config['lon']), elevation=float(psct_config['elevation']),
            catalog=sources_list
        )
    
    x_predicted = np.array([s['x'] for s in stars])
    y_predicted = np.array([s['y'] for s in stars])

    x_src_predicted = np.array([s['x'] for s in sources_pred])
    y_src_predicted = np.array([s['y'] for s in sources_pred])

    nearby_predicted = [(x, y) for x, y in zip(x_predicted, y_predicted) if np.min((x - x_detected)**2+(y - y_detected)**2) < distance_of_margin**2]

    found_turples = [(x, y) for x, y in zip(x_detected, y_detected)]

    sources_turple = [(x, y) for x, y in zip(x_src_predicted, y_src_predicted)]

    aa.MIN_MATCHES_FRACTION = min_matches_fraction
    k = max(len(nearby_predicted), len(found_turples))
    aa.NUM_NEAREST_NEIGHBORS = k+1
    aa.PIXEL_TOL = pixel_tol
    try:
        transf, (source_list, target_list) = aa.find_transform(nearby_predicted, found_turples, max_control_points = max_control_points, detection_sigma = detection_sigma, min_area = min_area)
        dst_calc = aa.matrix_transform(sources_turple, transf.params)

        for ind, x, y in zip(range(len(sources_pred)), [ob[0] for ob in dst_calc], [ob[1] for ob in dst_calc]):
            sources_pred[ind]['x'] = x
            sources_pred[ind]['y'] = y

        dst_calc = aa.matrix_transform([(x, y) for x, y in zip(x_predicted, y_predicted)], transf.params)
        for ind, x, y in zip(range(len(stars)), [ob[0] for ob in dst_calc], [ob[1] for ob in dst_calc]):
            stars[ind]['x'] = x
            stars[ind]['y'] = y


        deltas, rot_ang = transf.translation, transf.rotation
        delta_x, delta_y = deltas
        return a, objects, x_detected, y_detected, stars, x_predicted, y_predicted, sources_pred, delta_x, delta_y, rot_ang, transf, nearby_predicted, found_turples 
    except:
        print(f'Finding a match failed!')
        for ind, x, y in zip(range(len(sources_pred)), [ob for ob in x_src_predicted], [ob for ob in y_src_predicted]):
            sources_pred[ind]['x'] = x
            sources_pred[ind]['y'] = y

        for ind, x, y in zip(range(len(stars)), [ob for ob in x_predicted], [ob for ob in y_predicted]):
            stars[ind]['x'] = x
            stars[ind]['y'] = y
        return a, objects, x_detected, y_detected, stars, x_predicted, y_predicted, sources_pred, None, None, None, None, nearby_predicted, found_turples 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", help="run to analyse", default=None)
    parser.add_argument("-srs", "--subruns", help="The highest subrun to analyse", default="0")
    parser.add_argument("-si", "--save_plots", help="Whether the script saves images of the plots", action='store_true')
    parser.add_argument("-rc", "--recollect", help="Whether the script re-collects the r1 data stats", action='store_true')
    parser.add_argument("-config", "--configurations", help="path to config file", default='./settings.yaml')
    args = parser.parse_args()

    #### Settings loaded
    with open(args.configurations, 'r') as ymlfile:
        dval = yaml.safe_load(ymlfile)

    bad_srs = dval['bad_srs']
    ani_path = dval['analysis_options']['animation_path']
    npy_paths = dval['analysis_options']['npys_path']
    point_data_path = dval['analysis_options']['point_data_path']
    P_0 = dval['analysis_options']['P_step']
    dpi_imgs = dval['analysis_options']['dpi_imgs']
    log_files = dval['analysis_options']['log_files']
    subrun_files = dval['analysis_options']['subrun_files']
    catalog_path = dval['analysis_options']['catalog_path']
    telescope_config = dval['analysis_options']['telescope_config']
    run_info_path = dval['analysis_options']['run_info_path']


    ##### 

    run = int(args.run) # Will be an argument for a python calleable script
    ra_center, dec_center = fetch_pointing(int(run), run_info_path=run_info_path)

    if (not os.path.exists(f'{npy_paths}wfs_mean_run{run}.npy')) or args.recollect:
        # if os.path.exists(f'{npy_paths}wfs_mean_run{run}.npy'):
        #     print("Whaaaa")
        # if args.recollect:
        #     print("WROOOOONG")
        st = tm.time()
        first_collected = False

        for sr in sorted([int(get_value_from_filename(fil, '_subrun', '_r1.tio')) for fil in glob.glob(subrun_files.format(run, '*'))]):
            if run not in bad_srs.keys():
                bad_srs[run] = []
            if sr not in bad_srs[run]:
                reader = target_io.WaveformArrayReader(subrun_files.format(run, sr)) # Load reader object

                wfs_me0, wfs_sq0, times0 = read_wfs_metrics(None, reader = reader) # Calculate the waveforms means and mean of the squares, per pixel per event (this filters events with strong charges out)
                if first_collected:
                    wfs_me = np.append(wfs_me, wfs_me0, axis = 0)
                    wfs_sq = np.append(wfs_sq, wfs_sq0, axis = 0)
                    times = np.append(times, times0, axis = 0)
                else:
                    wfs_me = wfs_me0
                    wfs_sq = wfs_sq0
                    times = times0
                    first_collected = True


                if sr%5 == 0:
                    np.save(f'{npy_paths}wfs_mean_run{run}_temp.npy', wfs_me)
                    np.save(f'{npy_paths}wfs_mean_sq_run{run}_temp.npy', wfs_sq)
                    np.save(f'{npy_paths}wfs_times_run{run}_temp.npy', times)
        print(f'TIME INFO: Collecting all the data for run {run} took {(tm.time() - st)/60} mins')
        st = tm.time()
        np.save(f'{npy_paths}wfs_mean_run{run}.npy', wfs_me)
        np.save(f'{npy_paths}wfs_mean_sq_run{run}.npy', wfs_sq)
        np.save(f'{npy_paths}wfs_times_run{run}.npy', times)
        os.system(f'rm {npy_paths}wfs_mean_run{run}_temp.npy')
        os.system(f'rm {npy_paths}wfs_mean_sq_run{run}_temp.npy')
        os.system(f'rm {npy_paths}wfs_times_run{run}_temp.npy')
        print(f'TIME INFO: Saving all the data for run {run} took {(tm.time() - st)/60} mins')

    else:
        st = tm.time()
        wfs_me = np.load(f'{npy_paths}wfs_mean_run{run}.npy')
        wfs_sq = np.load(f'{npy_paths}wfs_mean_sq_run{run}.npy')
        times = np.load(f'{npy_paths}wfs_times_run{run}.npy')
        print(f'TIME INFO: Loading all the data for run {run} took {(tm.time() - st)/60} mins')

    psct_config = load_config(telescope_config) # Gets information about the telescope, like longitude, latitude, elevation, FoV
    catalog = load_hyg_catalog(catalog_path, mag_limit=10, named_only=False) # Loads star catalog

    st = tm.time()
    frames = []
    time_strs = []
    arg = 0
    arg_l = 0
    P = P_0*1E9
    base_timestamp = get_time_base(run, log_files)
    # base_timestamp = 1776574920.7772014
    timess = [int(t) for t in times]
    abs_times = []
    # print(timess[0])
    for ind, t in enumerate(timess):
        if (t - timess[arg_l]) > P:
            arg = arg_l
            arg_l = ind

            frames.append(get_gbmean([((wfs_sq_on - wfs_me_on**2)) for wfs_sq_on, wfs_me_on in zip(wfs_sq[arg:arg_l], wfs_me[arg:arg_l])], None).reshape(1600 // 64, 64))

            delta_t = (np.mean(timess[arg:arg_l+1]) - timess[0])/(1E9)
            timestamp_ev = convert_to_utc_str(base_timestamp + delta_t)
            abs_times.append(base_timestamp + delta_t)
            time_strs.append(timestamp_ev)

    frames_arr = [make_grid_camera(fr, fpms=fpms_t) for fr in frames]
    frames_arr = np.array(frames_arr)


    frames_med = frames_arr.copy()
    frames_med[frames_med == 0.0] = np.nan
    # frames_med[frames_med == 0.0] = np.nan
    # frames_med.sort(axis=0)
    frames_med = np.nanmedian(frames_med, axis = 0)

    frames_med = frames_med/np.nanmean(frames_med)
    frames_med[np.isnan(frames_med)] = 0.0
    fra_mean_ref = frames_arr.copy()
    fra_mean_ref[fra_mean_ref == 0.0] = np.nan

    fra_plot = [np.array([fra - np.nanmean(fra_m)*frames_med for fra, fra_m in zip(frames_arr, fra_mean_ref)])]
    bkg_base = np.load('./background.npy')
    list_frames = []
    for fra0 in fra_plot[0]:
        fra = fra0.copy()
        fra[fra == np.nan] = 0.0
        a0 = fra
        a = a0 # bkg_base*6.5 # np.nanmean(a0)
        # list_frames.append(np.array(a))
        list_frames.append(np.array(a))

    if args.save_plots:
        camdisp = CameraDisplay()

        camdisp.animate(list_frames, interval = 200, plot_title=f'Star Movie for Run {run}\n{convert_to_utc_str(base_timestamp)} UTC')
        camdisp.save(f'{ani_path}run{run}_star-movie.gif')

    print(f'TIME INFO: Making all the frames for run {run} took {(tm.time() - st)/60} mins')
    save = args.save_plots
    os.system(f'mkdir -p {point_data_path}run{run}_star_matched')
    if save: os.system(f'mkdir -p {point_data_path}run{run}_star_matched/full_frame_camera')
    # save = args.save_plots

    st = tm.time()
    dict_data = {
        'name': [],
        'x': [],
        'y': [],
        'x_mm': [],
        'y_mm': [],
        'time_utc': [],
        'time_abs': []
    }

    offset_data = {
        'x_c': [],
        'y_c': [],
        'x_c_mm': [],
        'y_c_mm': [],
        'time_utc': [],
        'time_abs': []
    }

    matrices = []
    frames = []
    pixel_outline = False
    module_outline = False
    sector_outline = False
    size = 5

    camdisp = CameraDisplay()
    for a0, time in zip(list_frames, abs_times):
        time_str = convert_to_utc_str(time)
        a, objects, x_detected, y_detected, stars, x_predicted, y_predicted, dst_calc, delta_x, delta_y, rot_ang, transf, nearby_predicted, found_turples = get_star_parameters_from_subrun_physical(a0, time_str, ra_center, dec_center, psct_config, catalog, sources_of_interest,
            thresh = 0.5, filter_type='matched', minarea = 2, deblend_nthresh=32, deblend_cont=0.005, 
            clean=False, clean_param=1.0, segmentation_map=False, distance_of_margin = 45, 
            min_matches_fraction = 0.2, pixel_tol = 13, max_control_points = 50, detection_sigma = 2, min_area = 3
        )

        # im = ax.imshow(a, interpolation=None, origin = 'lower', vmin = 0, vmax = 5, extent = [-60, 60, -60, 60])
        camdisp = CameraDisplay(rotate(a[:, ::-1], nturns = 1))
        if save:
            fig, ax = camdisp.plot(cbar_label = r'$\left< \text{Var} \right> \text{ } [\text{ADC}^2 \text{ns}^2]$')

            
            if pixel_outline:
                # Draw each pixel 
                for i in range(-60, a.shape[1]-60, 1):
                    for j in range(-60, a.shape[1]-60, 1):
                        ax.add_patch(mpl.patches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            linewidth=0.2*(size/8), edgecolor="lightgray", facecolor="none"
                        ))

            if module_outline:
                # Draw 8x8 block outlines (light gray)
                for i in range(-60, a.shape[1]-60, 8):
                    for j in range(-60, a.shape[1]-60, 8):
                        ax.add_patch(mpl.patches.Rectangle(
                            (j - 0.5, i - 0.5), 8, 8,
                            linewidth=0.8*(size/8), edgecolor="lightgray", facecolor="none"
                        ))

            if sector_outline:
                # Draw 40x40 block outlines (light black)
                for i in range(-60, a.shape[1]-60, 40):
                    for j in range(-60, a.shape[1]-60, 40):
                        ax.add_patch(mpl.patches.Rectangle(
                            (j - 0.5, i - 0.5), 40, 40,
                            linewidth=1.2*(size/8), edgecolor="#555555", facecolor="none"
                        ))
            # Plotting markers for detected stars
            for x, y, flu in zip(x_detected, y_detected, [obj[21] for obj in objects]):

                # ax.plot(x, y, marker = "x", color = 'red', markersize = 7*np.sqrt(flu/75))
                # ax.plot(pixel_to_length(x), pixel_to_length(y), marker = "x", color = 'red', markersize = 7*np.sqrt(flu/75))
                turp = transforming((x, y)) # transforming((pixel_to_length(x), pixel_to_length(y)))

                ax.plot(turp[0], turp[1], marker = "x", color = 'red', markersize = 7*np.sqrt(flu/75))



            # Plotting markers for possible stars in the field
            for x, y, flu, name in zip([s['x'] for s in stars], [s['y'] for s in stars], [10**(-s['mag']/2.5) for s in stars], [s['name'] for s in stars]):
                turp = transforming((x, y)) #  transforming((pixel_to_length(x), pixel_to_length(y)))
                if np.min((x - x_detected)**2+(y - y_detected)**2) < 15:
                    # turp = transforming((pixel_to_length(x), pixel_to_length(y)))
                    # ax.plot(x, y, marker = "*", color = 'yellow', alpha = 0.6)#, markersize = 7*np.sqrt(flu/75))

                    ax.plot(turp[0], turp[1], marker = "+", color = 'yellow', alpha = 0.6)#, markersize = 7*np.sqrt(flu/75))
                    if name in ['49    UMa']:
                        # ax.text(x, y, f'{name}', fontsize = 10)
                        # ax.text(pixel_to_length(x), pixel_to_length(y), f'{name}', fontsize = 10)


                        
                        ax.text(turp[0], turp[1], f'{name}', fontsize = 5)

                else:
                    ax.plot(turp[0], turp[1], marker = "*", color = 'yellow', alpha = 0.1)
        if delta_x == None:
            if save:
                for turp0 in nearby_predicted:
                    turp = transforming(turp0) #  transforming((pixel_to_length(x), pixel_to_length(y)))
                        # turp = transforming((pixel_to_length(x), pixel_to_length(y)))
                        # ax.plot(x, y, marker = "*", color = 'yellow', alpha = 0.6)#, markersize = 7*np.sqrt(flu/75))

                    ax.plot(turp[0], turp[1], marker = "x", color = 'green', alpha = 1, markersize = 10)

        if delta_x != None:
            matrices.append(transf.params)
            frames.append(a)
            if save:
                for x, y, name in zip([ob['x'] for ob in dst_calc], [ob['y'] for ob in dst_calc], [ob['name'] for ob in dst_calc]):
                    # ax.plot(x, y, marker = "+", color = "#01FFEE", markersize = 10)
                    # ax.text(x, y, f'{name}', fontsize = 15)

                    # ax.plot(pixel_to_length(x), pixel_to_length(y), marker = "+", color = "#01FFEE", markersize = 10)
                    # ax.text(pixel_to_length(x), pixel_to_length(y), f'{name}', fontsize = 15)

                    turp = transforming((x, y)) #  transforming((pixel_to_length(x), pixel_to_length(y)))


                    ax.plot(turp[0], turp[1], marker = "+", color = "#01FFEE", markersize = 10)
                    text_mrk = ax.text(turp[0], turp[1], f'{name}', fontsize = 15)
                    text_mrk.set_path_effects([fx.Stroke(linewidth=1, foreground='1.0'), fx.Normal()])

            turp_px = transforming((length_to_pixel(dst_calc[0]['x']), length_to_pixel(dst_calc[0]['y']))) # transforming((dst_calc[0]['x'], dst_calc[0]['y']))
            turp_mm = transforming((dst_calc[0]['x'], dst_calc[0]['y'])) # transforming((pixel_to_length(dst_calc[0]['x']), pixel_to_length(dst_calc[0]['y'])))

            dict_data['name'].append('Mrk 421')
            dict_data['x'].append(turp_px[0])
            dict_data['y'].append(turp_px[1])
            dict_data['x_mm'].append(turp_mm[0])
            dict_data['y_mm'].append(turp_mm[1])
            dict_data['time_abs'].append(time)
            dict_data['time_utc'].append(time_str)

            turp_cen_px = transforming((length_to_pixel(delta_x), length_to_pixel(delta_y))) # transforming((delta_x, delta_y))
            turp_cen_mm = transforming((delta_x, delta_y)) # transforming((pixel_to_length(delta_x), pixel_to_length(delta_y)))

            offset_data['x_c'].append(turp_cen_px[0])
            offset_data['y_c'].append(turp_cen_px[1])
            offset_data['x_c_mm'].append(turp_cen_mm[0])
            offset_data['y_c_mm'].append(turp_cen_mm[1])
            offset_data['time_abs'].append(time)
            offset_data['time_utc'].append(time_str)

            if save:
                # cbar = plt.colorbar(im, ax=ax,)
                # cbar.set_label(r'$\left< \text{Var} \right> \text{ } [\text{ADC}^2 \text{ns}^2]$')
                ax.set_title(f"Run {run}\n {time_str} UTC\n" + r"$\Delta x \equal$" +f"{turp_cen_mm[0]:.2f} mm, " + r"$\Delta y \equal$" +f"{turp_cen_mm[1]:.2f} mm, " + r"$\Delta \theta \equal$" +f"{rot_ang:.4f}", fontsize = 'x-small')
                ax.set_xlabel('CORSIKA x-axis [mm]')
                ax.set_ylabel('CORSIKA y-axis [mm]')
                # ax.set_xlim(-20.5, 19.5)
                # ax.set_ylim(-20.5, -60.5)
                fig.set_dpi(dpi_imgs)
                fig.savefig(f"{point_data_path}run{run}_star_matched/full_frame_camera/run{run}_{time_str.replace(':','-')}.jpeg")
                ax.set_xlim(-403, -128)
                ax.set_ylim(-138, 138)
                fig.savefig(f"{point_data_path}run{run}_star_matched/run{run}_{time_str.replace(':','-')}_sector7.jpeg")
        # plt.show()
        else:
            if save: fig.savefig(f"{point_data_path}run{run}_star_matched/full_frame_camera/run{run}_failed-frame_{time_str.replace(':','-')}.jpeg")
        plt.close()

    pd.DataFrame.from_dict(dict_data).to_csv(f'{point_data_path}run{run}_star_matched/run{run}_positions_mrk421.csv')
    pd.DataFrame.from_dict(offset_data).to_csv(f'{point_data_path}run{run}_star_matched/run{run}_center_offsets.csv')
    np.save(f'{point_data_path}run{run}_star_matched/run{run}_transformation_matrices.npy', np.array(matrices))
    np.save(f'{point_data_path}run{run}_star_matched/run{run}_image_frames.npy', np.array(frames))

    ext_str = 'out'
    if save: ext_str = ''
    print(f'TIME INFO: Fitting all the frames for run {run} took {(tm.time() - st)/60} mins with{ext_str} saving plots')

    if save:
        fig, ax = plt.subplots()
        # for i in range(-60, a.shape[1]-60, 40):
        #     for j in range(-60, a.shape[1]-60, 40):
        #         ax.add_patch(mpl.patches.Rectangle(
        #             (j - 0.5, i - 0.5), 40, 40,
        #             linewidth=1.2, edgecolor="#000000", facecolor="none"
        #         ))
        # for i in range(-60, a.shape[1]-60, 8):
        #     for j in range(-60, a.shape[1]-60, 8):
        #         ax.add_patch(mpl.patches.Rectangle(
        #             (j - 0.5, i - 0.5), 8, 8,
        #             linewidth=0.8*(size/8), edgecolor="lightgray", facecolor="none"
        #         ))
        base_frame = make_grid_camera([[]], fpms=[])
        grid_size = (float(psct_config['module_width'])/8)
        space = (grid_size - float(psct_config['pixel_size']))/2
        st = tm.time()
        for l in range(base_frame.shape[0]):
            for k in range(base_frame.shape[1]):
                if not np.isnan(base_frame[k, l]):
                    ax.add_patch(mpatches.Rectangle((pixel_to_length(l-60, float(psct_config['module_width']), float(psct_config['module_pitch']))+space, pixel_to_length(k-60, float(psct_config['module_width']), float(psct_config['module_pitch']))+space), float(psct_config['pixel_size']), float(psct_config['pixel_size']), edgecolor="#41414170", facecolor="#4141412A", alpha = 0.15))
        sc = ax.scatter(dict_data['x_mm'], dict_data['y_mm'], c = (np.array(dict_data['time_abs'])-dict_data['time_abs'][0])/(60), marker = 'o', cmap='viridis')
        ax.set_xlabel('CORSIKA x-axis [mm]')
        ax.set_ylabel('CORSIKA y-axis [mm]')
        ax.set_title(f'Mrk 421 camera path for run {run} with P = {P/1E9}s')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"Time from {dict_data['time_utc'][0]} [min]")
        plt.axis('scaled')
        ax.set_xlim(min((min(dict_data['x_mm']) - 20), -420), max((max(dict_data['x_mm']) + 20), -110))
        ax.set_ylim(min((min(dict_data['y_mm']) - 20), -160), max((max(dict_data['y_mm']) + 20), 160))
        fig.savefig(f"{point_data_path}run{run}_star_matched/run{run}_targets-trajectory.jpeg")
        print(f'Trajectory plot took {tm.time() - st} seconds')
        plt.close()

if __name__ == "__main__":
    main()