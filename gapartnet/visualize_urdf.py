import os
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from pathlib import Path
import math
import argparse

class URDFVisualizer:
    def __init__(self, urdf_path, width=800, height=600):
        self.urdf_path = Path(urdf_path)
        self.width = width
        self.height = height
        
        # 初始化Sapien引擎和场景
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 100.0)
        
        # 设置光照
        self._setup_lighting()
        
        # 加载URDF模型
        self.robot = self._load_urdf()
        
        # 设置相机
        self.camera = self._setup_camera()
        
        # 创建查看器 - 更新为 sapien.Viewer
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.toggle_axes(True)
    
    def _setup_lighting(self):
        """设置场景光照"""
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
    
    def _load_urdf(self):
        """加载URDF模型"""
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        
        # 配置URDF加载选项
        urdf_config = {
            "link_physx_material": {
                "static_friction": 0.5,
                "dynamic_friction": 0.5,
                "restitution": 0.0
            },
            "density": 1000
        }
        
        robot = loader.load(str(self.urdf_path), urdf_config)
        if not robot:
            raise ValueError(f"Failed to load URDF from {self.urdf_path}")
        return robot
    
    def _setup_camera(self, position=None, target=None):
        """设置相机位置和视角"""
        if position is None:
            position = np.array([3, 0, 1])  # 默认相机位置
        if target is None:
            target = np.array([0, 0, 0])    # 默认看向原点
        
        # 创建相机挂载点
        camera_mount = self.scene.create_actor_builder().build_kinematic()
        
        # 添加相机
        camera = self.scene.add_mounted_camera(
            name="main_camera",
            actor=camera_mount,
            pose=sapien.Pose(),  # 相对于挂载点的位姿
            width=self.width,
            height=self.height,
            fovy=np.deg2rad(45),
            near=0.1,
            far=100
        )
        
        # 设置相机位置和朝向
        self.set_camera_pose(camera_mount, position, target)
        return camera
    
    def set_camera_pose(self, mount, position, target=None):
        """设置相机位姿"""
        if target is None:
            target = np.zeros(3)
        
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        # 计算相机的左右和上方向
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        
        # 构建变换矩阵
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = position
        
        mount.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    
    def get_cam_pos(self, theta_min, theta_max, phi_min, phi_max, dis_min, dis_max):
        """随机生成相机位置"""
        theta = np.random.uniform(low=theta_min, high=theta_max)
        phi = np.random.uniform(low=phi_min, high=phi_max)
        distance = np.random.uniform(low=dis_min, high=dis_max)
        
        x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
        y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
        z = math.cos(math.pi / 180 * theta) * distance
        return np.array([x, y, z])
    
    def set_joint_positions(self, joint_qpos_dict):
        """设置关节位置"""
        for joint in self.robot.get_joints():
            joint_name = joint.get_name()
            if joint_name in joint_qpos_dict:
                joint.set_drive_target(joint_qpos_dict[joint_name])
    
    def capture_image(self):
        """捕获当前视图的图像"""
        self.scene.update_render()
        self.camera.take_picture()
        rgba = self.camera.get_float_texture('Color')
        rgb = rgba[..., :3]
        return (rgb * 255).clip(0, 255).astype("uint8")
    
    def run(self):
        """运行可视化"""
        # 设置初始相机视角
        self.viewer.set_camera_xyz(3, 0, 1)
        self.viewer.set_camera_rpy(0, -0.5, 0)
        
        print("Press 'q' to quit")
        while not self.viewer.closed:
            self.scene.step()
            self.scene.update_render()
            self.viewer.render()
        
        self.viewer.close()
# class URDFVisualizer:
#     def __init__(self, urdf_path, width=800, height=600):
#         self.urdf_path = Path(urdf_path)
#         self.width = width
#         self.height = height
#
#         # 初始化Sapien引擎和场景
#         self.engine = sapien.Engine()
#         self.renderer = sapien.SapienRenderer()
#         self.engine.set_renderer(self.renderer)
#         self.scene = self.engine.create_scene()
#         self.scene.set_timestep(1 / 100.0)
#
#         # 设置光照
#         self._setup_lighting()
#
#         # 加载URDF模型
#         self.robot = self._load_urdf()
#
#         # 创建查看器
#         self.viewer = Viewer(self.renderer)
#         self.viewer.set_scene(self.scene)
#         self.viewer.toggle_axes(True)
#
#         # 自动设置相机视角
#         self._auto_setup_camera()
#
#     def _setup_lighting(self):
#         """设置场景光照（保持不变）"""
#         self.scene.set_ambient_light([0.5, 0.5, 0.5])
#         self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
#         self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
#         self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
#         self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
#
#     def _load_urdf(self):
#         """加载URDF模型（保持不变）"""
#         loader = self.scene.create_urdf_loader()
#         loader.fix_root_link = True
#
#         urdf_config = {
#             "link_physx_material": {
#                 "static_friction": 0.5,
#                 "dynamic_friction": 0.5,
#                 "restitution": 0.0
#             },
#             "density": 1000
#         }
#
#         robot = loader.load(str(self.urdf_path), urdf_config)
#         if not robot:
#             raise ValueError(f"Failed to load URDF from {self.urdf_path}")
#         return robot
#
#     def _calculate_model_bbox(self):
#         """改进后的包围盒计算方法"""
#         min_bound = np.full(3, np.inf)
#         max_bound = np.full(3, -np.inf)
#
#         for link in self.robot.get_links():
#             # 同时考虑视觉和碰撞体
#             for shape in link.get_visual_shapes() + link.get_collision_shapes():
#                 geometry = shape.geometry
#                 pose = link.pose.transform(shape.get_local_pose())
#
#                 if isinstance(geometry, sapien.BoxGeometry):
#                     half_size = np.array(geometry.half_lengths)
#                     # 计算包围盒顶点的世界坐标
#                     vertices = np.array([
#                         [-1, -1, -1], [1, 1, 1]
#                     ]) * half_size
#                     world_vertices = pose.transform(vertices)
#                     min_bound = np.minimum(min_bound, world_vertices.min(axis=0))
#                     max_bound = np.maximum(max_bound, world_vertices.max(axis=0))
#
#                 elif isinstance(geometry, sapien.SphereGeometry):
#                     center = pose.p
#                     radius = geometry.radius
#                     min_bound = np.minimum(min_bound, center - radius)
#                     max_bound = np.maximum(max_bound, center + radius)
#
#                 elif isinstance(geometry, sapien.CapsuleGeometry):
#                     half_length = geometry.half_length
#                     radius = geometry.radius
#                     axis = pose.to_transformation_matrix()[:3, 2]  # Z轴方向
#                     tip1 = pose.p + axis * half_length
#                     tip2 = pose.p - axis * half_length
#                     min_bound = np.minimum(min_bound, np.minimum(tip1, tip2) - radius)
#                     max_bound = np.maximum(max_bound, np.maximum(tip1, tip2) + radius)
#
#         # 添加容错机制
#         if np.any(np.isinf(min_bound)) or np.any(np.isinf(max_bound)):
#             print("Warning: Invalid bounding box, using default")
#             return np.array([-1, -1, -1]), np.array([1, 1, 1])
#
#         return min_bound, max_bound
#
#     def _auto_setup_camera(self):
#         """改进后的相机定位方法"""
#         try:
#             bbox_min, bbox_max = self._calculate_model_bbox()
#             center = (bbox_min + bbox_max) / 2
#             size = bbox_max - bbox_min
#
#             # 自动检测主方向（假设最长边为正面方向）
#             main_axis = np.argmax(size)
#             front_directions = [
#                 np.array([1, 0, 0]),   # X轴
#                 np.array([0, 1, 0]),   # Y轴
#                 np.array([0, 0, 1])    # Z轴
#             ]
#             front_dir = front_directions[main_axis]
#
#             # 动态调整距离
#             distance = max(np.linalg.norm(size)*1.5, 1.0)
#             camera_pos = center - front_dir * distance + np.array([0, 0, distance*0.3])
#
#             # 设置相机参数
#             self.viewer.set_camera_xyz(*camera_pos)
#             self.viewer.set_camera_rpy(0, -0.4, 0)
#             self.viewer.window.set_camera_parameters(near=0.1, far=1000, fovy=45)  # 使用角度制
#         except Exception as e:
#             print(f"Camera setup failed: {e}, using default")
#             self.viewer.set_camera_xyz(3, 0, 1)
#             self.viewer.set_camera_rpy(0, -0.5, 0)
#
#     def run(self):
#         """（保持不变）"""
#         print("Press 'q' to quit")
#         while not self.viewer.closed:
#             self.scene.step()
#             self.scene.update_render()
#             self.viewer.render()
#         self.viewer.close()
def main():
    parser = argparse.ArgumentParser(description="URDF Visualizer using Sapien")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")
    args = parser.parse_args()
    urdf_path = "Refrigerator_11709_0_18_articulation.urdf"
    visualizer = URDFVisualizer(urdf_path, args.width, args.height)

    visualizer.run()

if __name__ == "__main__":
    main()