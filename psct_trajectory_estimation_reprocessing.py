from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import time as tm
import pandas as pd
import yaml
from psct_starfinder_pointing import load_config, make_grid_camera, pixel_to_length

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scaling: bool = True):
    """
    Oris had help of Claude AI when writing this function.
    Least-squares estimate of the similarity transform T(p) = s * R @ p + t
    that maps `src` points onto `dst` points, i.e. minimizes
        sum_i || dst_i - (s * R @ src_i + t) ||^2

        
    Parameters
    ----------
    src, dst : (K, 2) arrays of CORRESPONDING points (src[i] <-> dst[i])
    with_scaling : if False, forces s = 1 (rigid transform only)

    Returns
    -------
    s : float, isotropic scale
    R : (2,2) rotation matrix
    t : (2,) translation vector
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 2, "src/dst must be (K,2) and same shape"
    n = src.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 corresponding points to fit a similarity transform.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1  # correct for reflection

    R = U @ S @ Vt

    if with_scaling:
        var_src = (src_c ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    return s, R, t


def apply_transform(points_xy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply T(p) = s*R@p + t to an array of (K,2) points."""
    return (s * (R @ points_xy.T)).T + t


def load_config(filepath): # Function to load a config file. Should change to yaml at some point
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
    return result

def load_sources_file(path):
    result = pd.read_csv(path)
    result_form =  [
        {ke: float(result[ke][i]) if ke != 'name' else result[ke][i] for ke in result.columns} for i in range(len(result))
    ]
    return result_form

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", help="run to analyse", default=None)
    parser.add_argument("-srs", "--subruns", help="The highest subrun to analyse", default="0")
    parser.add_argument("-si", "--save_plots", help="Whether the script saves images of the plots", action='store_true')
    parser.add_argument("-rc", "--recollect", help="Whether the script re-collects the r1 data stats", action='store_true')
    parser.add_argument("-config", "--configurations", help="path to config file", default='./settings.yaml')
    parser.add_argument("-sources", "--sources_of_interest", help="path to sources of interest", default='./sources.csv')
    args = parser.parse_args()

    #### Settings loaded
    with open(args.configurations, 'r') as ymlfile:
        dval = yaml.safe_load(ymlfile)


    point_data_path = dval['analysis_options']['point_data_path']
    P_0 = dval['analysis_options']['P_step']

    telescope_config = dval['analysis_options']['telescope_config']
    scale_factor_bounds = dval['analysis_options']['bounds_scale_factor']
    rot_ang_bounds = dval['analysis_options']['bounds_rotation_angle']
    psct_config = load_config(telescope_config) # Gets information about the telescope, like longitude, latitude, elevation, FoV
    sources_of_interest = load_sources_file(args.sources_of_interest)


    ##### 

    run = int(args.run) # Will be an argument for a python calleable script

    run_info = pd.read_csv(f'{point_data_path}run{run}_star_matched/run{run}_positions_mrk421.csv')
    offset_info = pd.read_csv(f'{point_data_path}run{run}_star_matched/run{run}_center_offsets.csv')

    if len(run_info['x_mm']) > 0:
        nom_matched = np.array([[float(x), float(y)] for x, y, x_m, s, r in zip(run_info['x_mm_nominal'], run_info['y_mm_nominal'], run_info['x_mm'], offset_info['scale_factor'], offset_info['rot_ang']) if (not np.isnan(float(x_m)) and r > rot_ang_bounds[0] and r < rot_ang_bounds[1] and s > scale_factor_bounds[0] and s < scale_factor_bounds[1])])
        meas_matched = np.array([[float(x), float(y)] for x, y, s, r in zip(run_info['x_mm'], run_info['y_mm'], offset_info['scale_factor'], offset_info['rot_ang']) if (not np.isnan(float(x)) and r > rot_ang_bounds[0] and r < rot_ang_bounds[1] and s > scale_factor_bounds[0] and s < scale_factor_bounds[1])]) # isinstance(float(x), (float))
        nom_all = np.array([[float(x), float(y)] for x, y in zip(run_info['x_mm_nominal'], run_info['y_mm_nominal'])])
        tim_abs_matched = [t for x_m, t, s, r in zip(run_info['x_mm'], run_info['time_abs'], offset_info['scale_factor'], offset_info['rot_ang']) if (not np.isnan(float(x_m)) and r > rot_ang_bounds[0] and r < rot_ang_bounds[1] and s > scale_factor_bounds[0] and s < scale_factor_bounds[1])]
        tim_utc_matched = [t for x_m, t, s, r in zip(run_info['x_mm'], run_info['time_utc'], offset_info['scale_factor'], offset_info['rot_ang']) if (not np.isnan(float(x_m)) and r > rot_ang_bounds[0] and r < rot_ang_bounds[1] and s > scale_factor_bounds[0] and s < scale_factor_bounds[1])]
        tim_abs_arr = [t for t in run_info['time_abs']]
        tim_utc_arr = [t for t in run_info['time_utc']]
        try:
            s, R, t = umeyama_similarity(nom_matched, meas_matched, with_scaling=True)
            est_list = apply_transform(nom_all, s=s, R=R, t=t)
            est_list_x = [l[0] for l in est_list]
            est_list_y = [l[1] for l in est_list]
            dict_estimated = {
                'name': [sources_of_interest[0]['name'] for _ in range(len(est_list_y))],
                'x_mm_estimated': est_list_x,
                'y_mm_estimated': est_list_y,
                'time_abs': tim_abs_arr,
                'time_utc': tim_utc_arr
            }
            pd.DataFrame.from_dict(dict_estimated).to_csv(f'{point_data_path}run{run}_star_matched/run{run}_estimated_positions_mrk421.csv')
        except:
            est_list_x = []
            est_list_y = []
            print(f'Estimated path failed to fit: only {len(meas_matched)} points passed the filter')


    # print(f'TIME INFO: Fitting all the frames for run {run} took {(tm.time() - st)/60} mins with{ext_str} saving plots')

    if len(sources_of_interest)>0:
        fig, ax = plt.subplots()

        base_frame = make_grid_camera([[]], fpms=[])
        grid_size = (float(psct_config['module_width'])/8)
        space = (grid_size - float(psct_config['pixel_size']))/2
        st = tm.time()
        for l in range(base_frame.shape[0]):
            for k in range(base_frame.shape[1]):
                if not np.isnan(base_frame[k, l]):
                    ax.add_patch(mpatches.Rectangle((pixel_to_length(l-60, float(psct_config['module_width']), float(psct_config['module_pitch']))+space, pixel_to_length(k-60, float(psct_config['module_width']), float(psct_config['module_pitch']))+space), float(psct_config['pixel_size']), float(psct_config['pixel_size']), edgecolor="#41414170", facecolor="#4141412A", alpha = 0.15))
        # if len(times_f) > 0:
        #     t_corr = float(times_f[0])
        # else:
        #     print(f'length is 0 and number of frames is {len(list_frames)}')
        #     t_corr = 0.0
        ax.plot(run_info['x_mm_nominal'], run_info['y_mm_nominal'], c='black', label = 'Nominal path', marker='o')
        sc = ax.scatter([x for x in run_info['x_mm'] if isinstance(x, (float))], [y for y in run_info['y_mm'] if isinstance(y, (float))], c = (np.array([t for x, t in zip(run_info['x_mm'], run_info['time_abs']) if  isinstance(x, (float))])-offset_info['time_abs'][0])/(60), marker = 'o', cmap='viridis', label = 'Measured path')
        if len(est_list_x) > 0:
            ax.plot(est_list_x, est_list_y, marker='+', markersize = 5, c='red', label = 'Estimated path')

        ax.set_xlabel('CORSIKA x-axis [mm]')
        ax.set_ylabel('CORSIKA y-axis [mm]')
        ax.set_title(f"{sources_of_interest[0]['name']} camera path for run {run} with P = {P_0}s")
        cbar = fig.colorbar(sc, ax=ax)
        ax.legend()
        cbar.set_label(f"Time from {offset_info['time_utc'][0]} [min]")
        plt.axis('scaled')
        if len(run_info['x_mm']) > 0:
            # print([x for x in run_info['x_mm'] if isinstance(x, (float))])
            ax.set_xlim(min((np.nanmin([x for x in run_info['x_mm'] if isinstance(x, (float))]) - 20), -420), max((np.nanmax([x for x in run_info['x_mm'] if isinstance(x, (float))]) + 20), -110))
            ax.set_ylim(min((np.nanmin([y for y in run_info['y_mm'] if isinstance(y, (float))]) - 20), -160), max((np.nanmax([y for y in run_info['y_mm'] if isinstance(y, (float))]) + 20), 160))
        else:
            ax.set_xlim(-420, -110)
            ax.set_ylim(-160, 160)
        fig.savefig(f"{point_data_path}run{run}_star_matched/run{run}_targets-trajectory.jpeg")
        print(f'Trajectory plot took {tm.time() - st} seconds')
        plt.close()


if __name__ == "__main__":
    main()