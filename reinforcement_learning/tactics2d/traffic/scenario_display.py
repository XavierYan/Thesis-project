##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: scenario_display.py
# @Description: This script is used to display the scenario with matplotlib.
# @Author: Yueyuan Li
# @Version: 1.0.0

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from tactics2d.map.element import Area, Lane, RoadLine
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.sensor.render_template import *
from utils_map import *


class ScenarioDisplay:
    """This class implements a matplotlib-based scenario visualizer."""

    def __init__(self):
        self.participant_patches = dict()
        self.trajectory_lines = dict()
        self.reference_trajectory_lines = dict()

    def _get_color(self, element):
        if element.color in COLOR_PALETTE:
            return COLOR_PALETTE[element.color]

        if element.color is None:
            if hasattr(element, "subtype") and element.subtype in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.subtype]
            if hasattr(element, "type_") and element.type_ in DEFAULT_COLOR:
                return DEFAULT_COLOR[element.type_]
            if isinstance(element, Area):
                return DEFAULT_COLOR["area"]
            if isinstance(element, Lane):
                return DEFAULT_COLOR["lane"]
            if isinstance(element, RoadLine):
                return DEFAULT_COLOR["roadline"]
            if isinstance(element, Vehicle):
                return DEFAULT_COLOR["vehicle"]
            if isinstance(element, Cyclist):
                return DEFAULT_COLOR["cyclist"]
            if isinstance(element, Pedestrian):
                return DEFAULT_COLOR["pedestrian"]

        if len(element.color) == 3 or len(element.color) == 4:
            if 1 < np.max(element.color) <= 255:
                return tuple([color / 255 for color in element.color])

    def _get_order(self, element):
        if hasattr(element, "subtype") and element.subtype in DEFAULT_ORDER:
            return DEFAULT_ORDER[element.subtype]
        if hasattr(element, "type_") and element.type_ in DEFAULT_ORDER:
            return DEFAULT_ORDER[element.type_]
        if isinstance(element, Area):
            return DEFAULT_ORDER["area"]
        if isinstance(element, Lane):
            return DEFAULT_ORDER["lane"]
        if isinstance(element, RoadLine):
            return DEFAULT_ORDER["roadline"]
        if isinstance(element, Vehicle):
            return DEFAULT_ORDER["vehicle"]
        if isinstance(element, Cyclist):
            return DEFAULT_ORDER["cyclist"]
        if isinstance(element, Pedestrian):
            return DEFAULT_ORDER["pedestrian"]

    def _get_line(self, roadline):
        if roadline.type_ == "virtual":
            return None

        line_shape = np.array(roadline.shape)
        line_width = 0.5 if roadline.type_ in ["line_thin", "curbstone"] else 1

        if roadline.subtype == "solid_solid":
            line1 = Line2D(
                line_shape[:, 0] + 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            line2 = Line2D(
                line_shape[:, 0] - 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            return [line1, line2]

        if roadline.subtype == "dashed":
            return [
                Line2D(
                    line_shape[:, 0],
                    line_shape[:, 1],
                    linewidth=line_width,
                    linestyle=(0, (5, 5)),
                    color=self._get_color(roadline),
                    zorder=self._get_order(roadline),
                )
            ]

        if roadline.subtype == "dashed_dashed":
            line1 = Line2D(
                line_shape[:, 0] + 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                linestyle=(0, (5, 5)),
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

            line2 = Line2D(
                line_shape[:, 0] - 0.05,
                line_shape[:, 1],
                linewidth=line_width,
                linestyle=(0, (5, 5)),
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )

        return [
            Line2D(
                line_shape[:, 0],
                line_shape[:, 1],
                linewidth=line_width,
                color=self._get_color(roadline),
                zorder=self._get_order(roadline),
            )
        ]

    def display_map(self, map_, ax):
        # ax.set_xlim(20, 150) # 设置x轴范围
        # ax.set_ylim(-85, -10)  # 设置y轴范围
        for area in map_.areas.values():
            area = Polygon(
                area.geometry.exterior.coords,
                True,
                facecolor=self._get_color(area),
                edgecolor=None,
                zorder=self._get_order(area),
            )
            ax.add_patch(area)

        for lane in map_.lanes.values():
            lane = Polygon(
                lane.geometry.coords,
                True,
                facecolor=self._get_color(lane),
                edgecolor=None,
                zorder=self._get_order(lane),
            )
            ax.add_patch(lane)

        for roadline in map_.roadlines.values():
            lines = self._get_line(roadline)
            if lines is None:
                continue

            for line in lines:
                ax.add_line(line)

    def update_participants(self, frame, participants, ax):
        ax.frame_text.set_text(f'Frame: {frame}')  # 直接使用ax的属性更新文本
        for participant in participants.values():
            try:
                participant.get_state(frame)
            except:
                if (
                    participant.id_ in self.participant_patches
                    and participant.trajectory.last_frame < frame
                ):
                    self.participant_patches[participant.id_].remove()
                    self.participant_patches.pop(participant.id_)

                continue

            if isinstance(participant, Vehicle):
                if participant.id_ not in self.participant_patches:
                    self.participant_patches[participant.id_] = ax.add_patch(
                        Polygon(
                            participant.get_pose(frame).coords,
                            True,
                            facecolor=self._get_color(participant),
                            edgecolor=None,
                            zorder=self._get_order(participant),
                        )
                    )
                else:
                    self.participant_patches[participant.id_].set_xy(
                        participant.get_pose(frame).coords
                    )

        return list(self.participant_patches.values())
    

    def update_participants_realtime(self, frame, participants, model, ll_map,ax):
        # for p in ax.patches:
        #     p.remove()

        # for line in list(ax.lines):
        #     line.remove()
        
        updated_objects = []  # 存储需要更新的图形对象
        ax.frame_text.set_text(f'Frame: {frame}')  # 直接使用ax的属性更新文本

        for participant in participants.values():
            # 使用模型预测控制信号和轨迹
            acceleration, steering_rate = participant.get_ctr(model,ll_map)
            new_state = participant.physics_model.step(participant.trajectory.last_state, acceleration, steering_rate)
            participant.trajectory.add_state(new_state)
            
            # 获取并绘制预测的轨迹
            trajectory = participant.get_traj(model, ll_map)
            ref_traj = participant.reference_trajectory
            x_traj, y_traj = zip(*trajectory)  # 解压轨迹的坐标
            x_ref, y_ref = zip(*ref_traj)
            if participant.id_ not in self.trajectory_lines:
                # 如果没有现有的轨迹线条对象，创建一个新的
                trajectory_line, = ax.plot(x_traj, y_traj, 'o', linestyle='None', color='green',markersize = 2,zorder=12)
                self.trajectory_lines[participant.id_] = trajectory_line
                ref_line, = ax.plot(x_ref, y_ref, '-', color='black', linewidth=1, markersize=3, zorder=10)
                self.reference_trajectory_lines[participant.id_] = ref_line
            else:
                # 更新现有的轨迹线条对象
                trajectory_line = self.trajectory_lines[participant.id_]
                trajectory_line.set_data(x_traj, y_traj)
                ref_line = self.reference_trajectory_lines[participant.id_]
                ref_line.set_data(x_ref, y_ref)
            
            updated_objects.append(trajectory_line) 
            updated_objects.append(ref_line)

            # 绘制车辆当前位置的多边形
            pose = participant.get_pose()
            if participant.id_ not in self.participant_patches:
                patch = ax.add_patch(
                    Polygon(
                        pose.coords,
                        True,
                        facecolor=self._get_color(participant),
                        edgecolor=None,
                        zorder=self._get_order(participant),
                    )
                )
                self.participant_patches[participant.id_] = patch
            else:
                patch = self.participant_patches[participant.id_]
                patch.set_xy(pose.coords)

            updated_objects.append(patch)  # 将多边形对象也添加到更新列表
        
        updated_objects.append(ax.frame_text)  # 确保文本也在更新列表中
        return updated_objects  # 返回所有需要更新的图形对象


    def display_realtime(self, participants, map_, interval, frames, fig_size,model = None, ll_map = None,**ax_kwargs):
        fig, ax = plt.subplots()
        ax.set_xlim(20, 150) # 设置x轴范围
        ax.set_ylim(-85, -10)  # 设置y轴范围
        ax.frame_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha="right", va="top")
        fig.set_size_inches(fig_size)
        fig.set_layout_engine("none")
        ax.set(**ax_kwargs)
        ax.set_axis_off()

        # if "womd" in map_.name:
        #     fig.set_facecolor(COLOR_PALETTE["black"])

        self.display_map(map_, ax)
        ax.plot()

        animation = FuncAnimation(
            fig,
            self.update_participants_realtime,
            frames=frames,
            fargs=(participants,model, ll_map, ax),
            # fargs=(participants,ax),
            interval=interval,
        )
        plt.show()
        return animation
    
    def display(self, participants, map_, interval, frames, fig_size,**ax_kwargs):
        fig, ax = plt.subplots()
        ax.set_xlim(20, 150) # 设置x轴范围
        ax.set_ylim(-85, -10)  # 设置y轴范围
        ax.frame_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha="right", va="top")
        fig.set_size_inches(fig_size)
        fig.set_layout_engine("none")
        ax.set(**ax_kwargs)
        ax.set_axis_off()

        # if "womd" in map_.name:
        #     fig.set_facecolor(COLOR_PALETTE["black"])

        self.display_map(map_, ax)
        ax.plot()

        animation = FuncAnimation(
            fig,
            self.update_participants,
            frames=frames,
            fargs=(participants,ax),
            interval=interval,
        )
        plt.show()
        return animation

    def reset(self):
        for patch in self.participant_patches.values():
            patch.remove()

        self.participant_patches.clear()
