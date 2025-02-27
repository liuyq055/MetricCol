from __future__ import absolute_import, division, print_function
from options import MonodepthOptions
import os

import numpy as np
import matplotlib.pyplot as plt
import evo.core.lie_algebra as lie

from evo.tools import plot, file_interface
from evo.core import units
from evo.core.units import Unit
import copy
from evo.core import metrics
import pprint
import logging
from evo.tools import plot, file_interface, log, settings

units.METER_SCALE_FACTORS[Unit.millimeters]=1

logger = logging.getLogger("evo")
log.configure_logging(verbose=True)


def save_kitti_traj(pred_pth, save=None):

    pred_poses = np.load(pred_pth)
    poses_mat = []
    cam_to_world = np.eye(4)
    poses_mat.append(cam_to_world[:3,:])
    for i in range(0, len(pred_poses)):
        cam_to_world = np.dot(cam_to_world, np.linalg.inv(pred_poses[i]))

        poses_mat.append(cam_to_world[:3,:])

    if save:
        with open(f'{save}', 'w') as f:
            # 遍历每个 4x4 的位姿矩阵
            for pose in poses_mat:
                # 将 4x4 矩阵展平为一个 16 维的向量
                flattened_pose = pose.flatten()
                pose_str = ' '.join(f"{x:.12f}" for x in flattened_pose)

                # 将展平的位姿以空格分隔的形式写入文件
                f.write(pose_str + '\n')
    return

def save_gt_poses(gt_path, save=None):
    """
    读取文件并转换为kitti格式
    """

    poses_mat = []
    traj = open(gt_path,'r')
    for line in traj:
        pose = np.array(list(map(float, line.split(',')))).reshape((4, 4))
        poses_mat.append(pose[:3,:])

    if save:
        with open(f'{save}', 'w') as f:
            # 遍历每个 4x4 的位姿矩阵
            for pose in poses_mat:
                # 将 4x4 矩阵展平为一个 16 维的向量
                flattened_pose = pose.flatten()
                pose_str = ' '.join(f"{x:.12f}" for x in flattened_pose)

                # 将展平的位姿以空格分隔的形式写入文件
                f.write(pose_str + '\n')

    return


def metrics_ape(traj_ref, traj_est, save_dir):
    max_diff = 0.01
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    import evaluate.ATE_metrics
    pose_relation = evaluate.ATE_metrics.PoseRelation.translation_part
    ate_metric = evaluate.ATE_metrics.ATE(pose_relation)
    ate_metric.process_data(data)
    ate_stats = ate_metric.get_all_statistics()
    pprint.pprint(ate_stats)

    ape_stats = ape_metric.get_all_statistics()
    pprint.pprint(ape_stats)
    seconds_from_start = range(0, traj_ref.num_poses)
    fig = plt.figure(figsize=(8, 8),dpi=1000)
    plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                     statistics={s: v for s, v in ape_stats.items() if s != "sse"},
                     name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
    # plt.show()
    fig.savefig(f"{save_dir}/ape_error.png")
    plt.close(fig)


    plot_mode = plot.PlotMode.xy
    fig = plt.figure(figsize=(8, 8),dpi=1000)
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est, ape_metric.error,
                       plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
    ax.legend()
    # plt.show()
    fig.savefig(f"{save_dir}/traj_color_map.png")
    plt.close(fig)

    return ape_stats, ate_stats


def metrics_rpe(traj_ref, traj_est, save_dir):

    pose_relation = metrics.PoseRelation.rotation_angle_deg

    # normal mode
    delta = 1
    delta_unit = Unit.frames

    # all pairs mode
    all_pairs = False  # activate

    data = (traj_ref, traj_est)
    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    rpe_metric.process_data(data)


    rpe_stats = rpe_metric.get_all_statistics()
    pprint.pprint(rpe_stats)

    import copy
    traj_ref_plot = copy.deepcopy(traj_ref)
    traj_est_plot = copy.deepcopy(traj_est)
    traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
    traj_est_plot.reduce_to_ids(rpe_metric.delta_ids)
    seconds_from_start = range(0, traj_ref.num_poses-1)

    fig = plt.figure(figsize=(8, 8),dpi=1000)
    plot.error_array(fig.gca(), rpe_metric.error, x_array=seconds_from_start,
                     statistics={s: v for s, v in rpe_stats.items() if s != "sse"},
                     name="RPE", title="RPE w.r.t. " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")
    # plt.show()

    fig.savefig(f"{save_dir}/rpe_error.png")
    plt.close(fig)

    plot_mode = plot.PlotMode.xy
    fig = plt.figure(figsize=(8, 8),dpi=1000)
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref_plot, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est, rpe_metric.error, plot_mode, min_map=rpe_stats["min"],
                       max_map=rpe_stats["max"])
    ax.legend()
    # plt.show()
    fig.savefig(f"{save_dir}/rpe_colormap.png")
    plt.close(fig)

    return rpe_stats



def evaluate_metric(opt, scene, visualize=True):
    logger = logging.getLogger("evo")
    log.configure_logging(verbose=True)

    save_dir = f'results/c3vd/poses/{scene}'

    gt_path = os.path.join(opt.data_path, scene, 'pose.txt')
    pred_pth = f'{save_dir}/{scene}_pred_poses.npy'
    assert os.path.exists(gt_path), f'No poses folder found  {gt_path}'

    save_gt_poses(gt_path, f'{save_dir}/{scene}_kitti_gt.txt')
    save_kitti_traj(pred_pth, f'{save_dir}/{scene}_kitti_est.txt')



    traj_ref = file_interface.read_kitti_poses_file(f'{save_dir}/{scene}_kitti_gt.txt')
    traj_est = file_interface.read_kitti_poses_file(f'{save_dir}/{scene}_kitti_est.txt')

    traj_est.transform(lie.se3(np.eye(3), np.array([0, 0, 0])))
    traj_est.scale(0.5)

    traj_est_aligned_scaled = copy.deepcopy(traj_est)
    traj_est_aligned_scaled.align(traj_ref, correct_scale=True)

    if visualize:

        fig = plt.figure(figsize=(8, 8),dpi=1000)
        plot_mode = plot.PlotMode.xyz

        ax = plot.prepare_axis(fig, plot_mode, subplot_arg=111, length_unit=units.Unit.millimeters)
        plot.traj(ax, plot_mode, traj_ref,  '--', 'red', )
        plot.traj(ax, plot_mode, traj_est_aligned_scaled, '-', 'blue')
        fig.axes.append(ax)
        # plt.show()
        fig.savefig(f'{save_dir}/{scene}_traj.png')
        plt.close(fig)

    ape, ate= metrics_ape(traj_ref, traj_est_aligned_scaled, save_dir)
    rpe = metrics_rpe(traj_ref, traj_est_aligned_scaled, save_dir)


    with open(f"{save_dir}/{scene}_error.txt", "w") as f:
        # 将 ATE 和 RE 的结果写入文件
        f.write(f"ape:, {ape} \n")
        f.write(f"rpe:, {rpe}\n")
        f.write(f"ate:, {ate}")


    return ape, rpe ,ate

if __name__ == "__main__":
    options = MonodepthOptions()
    test_scene = ["s3v2", "t3v1", "t4v1", "d4v1", "c4v1"]
    for scene in test_scene:
        evaluate_metric(options.parse(), scene=scene)