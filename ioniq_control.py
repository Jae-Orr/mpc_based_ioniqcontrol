#! /usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point, PoseStamped, Vector3
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry, Path
from novatel_oem7_msgs.msg import BESTGNSSPOS
from collections import deque
import numpy as np
from math import radians, degrees, atan2, cos, sin, sqrt
import tf
import math
from geopy.distance import geodesic
import time
import json

# Actuator 메시지 (vehicle_control.msg.Actuator)
from vehicle_control.msg import Actuator

# -------------------------------------------
# 전역 경로 정의 (x, y, yaw; yaw는 radian 단위)
waypoints = [
    # 섹션 0
    (37.386011, 126.652558, -1.05),
    (37.386027, 126.652530, -1.01),
    (37.386046, 126.652500, -1.01),
    (37.386066, 126.652467, -1.00),
    (37.386087, 126.652433, -0.98),
    (37.386109, 126.652396, -0.98),
    (37.386132, 126.652358, -0.97),
    (37.386156, 126.652317, -0.96),
    (37.386186, 126.652270, -0.96),
    # 섹션 1
    (37.386988, 126.648747, -2.38),
    (37.386944, 126.648705, -2.38),
    (37.386901, 126.648664, -2.38),
    (37.386859, 126.648623, -2.38),
    (37.386818, 126.648584, -2.38),
    (37.386776, 126.648543, -2.38),
    # 섹션 2
    (37.385401, 126.648788, 2.11),
    (37.385360, 126.648856, 2.11),
    (37.385322, 126.648918, 2.11),
    (37.385288, 126.648973, 2.11),
    (37.385254, 126.649029, 2.11),
    # 섹션 3
    (37.384455, 126.650347, 2.14),
    (37.384404, 126.650428, 2.14),
    (37.384354, 126.650510, 2.14),
    (37.384303, 126.650592, 2.14),
    # 섹션 4
    (37.383819, 126.651387, 2.11),
    (37.383778, 126.651455, 2.11),
    (37.383737, 126.651523, 2.11),
    (37.383716, 126.651557, 2.11),
    # 섹션 5
    (37.383053, 126.652649, 2.11),
    (37.383014, 126.652713, 2.11),
    (37.382981, 126.652767, 2.11),
    (37.382954, 126.652811, 2.11),
    # 섹션 6
    (37.382702, 126.653218, 2.11),
    (37.382683, 126.653249, 2.15),
    (37.382659, 126.653287, 2.13),
    (37.382637, 126.653323, 2.09),
    (37.382621, 126.653351, 2.09),
    (37.382611, 126.653376, 1.95),
    (37.382606, 126.653398, 1.79),
    (37.382606, 126.653430, 1.57),
    (37.382614, 126.653463, 1.33),
    (37.382632, 126.653498, 1.10),
    (37.382658, 126.653532, 0.92),
    (37.382681, 126.653555, 0.79),
    (37.382677, 126.653550, 0.79),
    (37.382705, 126.653578, 0.79),
    # 섹션 7
    (37.383053, 126.653905, 0.77),
    (37.383119, 126.653969, 0.77),
    (37.383170, 126.654019, 0.77),
    (37.383241, 126.654088, 0.77),
    # 섹션 8
    (37.383612, 126.654445, 0.75),
    (37.383697, 126.654525, 0.75),
    (37.383768, 126.654591, 0.77),
    (37.383836, 126.654656, 0.77)
]

waypoint_sections = {
    0: waypoints[0:9],
    1: waypoints[9:15],
    2: waypoints[15:20],
    3: waypoints[20:24],
    4: waypoints[24:28],
    5: waypoints[28:32],
    6: waypoints[32:46],
    7: waypoints[46:50],
    8: waypoints[50:58],
}

# -------------------------------------------
# Parameter, Node, PATH 클래스
# 상태벡터: [x, y, yaw]
# 제어 입력: [v, steer]  (v: m/s, steer: radian)
class Parameter:
    NX = 3      # [x, y, yaw]
    NU = 2      # [v, steer]
    T = 20      # 예측 시간 스텝
    dt = 0.2

    target_speed = 15.0  # km/h (참조용)
    speed_max = 20.0 / 3.6  # m/s
    speed_min = 0.0         # m/s
    steer_max = np.deg2rad(21.0)
    WB = 1.04  # wheel base

class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v  # 현재 속도 (m/s)
        self.direct = 1

    # 제어 입력 (v, steer)를 받아 상태 업데이트 (간단 자전거 모델)
    def update(self, v, steer, gear):
        self.x += v * math.cos(self.yaw) * Parameter.dt
        self.y += v * math.sin(self.yaw) * Parameter.dt
        self.yaw += v / Parameter.WB * math.tan(steer) * Parameter.dt
        self.v = v

class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)

    def nearest_index(self, node):
        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        dist = np.hypot(dx, dy)
        return int(np.argmin(dist))

# -------------------------------------------
# MPC 관련 함수 (상태: [x, y, yaw], 제어: [v, steer])
def calc_ref_trajectory_in_T_step(node, ind, ref_path, sp):
    z_ref = np.zeros((Parameter.NX, Parameter.T + 1))
    length = ref_path.length
    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = ref_path.cyaw[ind]
    z_ref[3, 0] = sp[ind]
    dist_move = 0.0
    for i in range(1, Parameter.T + 1):
        dist_move += node.v * Parameter.dt
        ind_move = int(round(dist_move / Parameter.d_dist))
        index = min(ind + ind_move, length - 1)
        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = ref_path.cyaw[index]
        z_ref[3, i] = sp[index]
    return z_ref

def mpc_predict_next_state(z_ref, x0, y0, yaw0, v, steer):
    z_bar = np.zeros((Parameter.NX, Parameter.T + 1))
    z_bar[0, 0] = x0
    z_bar[1, 0] = y0
    z_bar[2, 0] = yaw0
    for i in range(Parameter.T):
        x1 = x0 + v * math.cos(yaw0) * Parameter.dt
        y1 = y0 + v * math.sin(yaw0) * Parameter.dt
        yaw1 = yaw0 + v / Parameter.WB * math.tan(steer) * Parameter.dt
        z_bar[0, i+1] = x1
        z_bar[1, i+1] = y1
        z_bar[2, i+1] = yaw1
        x0, y0, yaw0 = x1, y1, yaw1
    return z_bar

def mpc_cost_function(z_ref, z_bar, steer_list):
    cost = np.sum((z_ref - z_bar)**2)
    return cost

# 후보 생성 함수들 (이 방식대로 후보 입력값을 생성)
def mpc_candidate_v(nominal_v, delta=0.2, num_candidates=5):
    candidate_vs = []
    for i in range(num_candidates):
        candidate = nominal_v - delta * (num_candidates // 2) + i * delta
        candidate_vs.append(candidate)
    return candidate_vs

def mpc_candidate_steer(node, target_point, base_steer, delta=1.0, num_candidates=5):
    candidate_steers = []
    for i in range(num_candidates):
        candidate = base_steer - delta * (num_candidates // 2) + i * delta
        candidate_steers.append(candidate)
    return candidate_steers

# -------------------------------------------
# Pure Pursuit (간단화)
class PurePursuit:
    def __init__(self):
        self.Lfc = 6.0
        self.k = 0.14

    def run(self, v, target_point, current_position, yaw, curr_steer):
        dx = target_point[0] - current_position[0]
        dy = target_point[1] - current_position[1]
        desired_angle = atan2(dy, dx)
        steer = desired_angle - yaw
        steer = max(min(steer, Parameter.steer_max), -Parameter.steer_max)
        return degrees(steer), target_point

# -------------------------------------------
# PID (종제어: 속도 제어)
class PID:
    def __init__(self, kp, ki, kd, dt=0.05):
        self.K_P = kp
        self.K_I = ki
        self.K_D = kd
        self.pre_error = 0.0
        self.integral_error = 0.0
        self.dt = dt

    def run(self, target, current):
        error = sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)
        derivative_error = (error - self.pre_error) / self.dt
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -5, 5)
        output = self.K_P * error + self.K_I * self.integral_error + self.K_D * derivative_error
        output = np.clip(output, 0, Parameter.speed_max)
        self.pre_error = error
        return output

# -------------------------------------------
# Start 클래스: ROS 통신, 센서 업데이트, 시나리오 유지 (글로벌 & 로컬 제어)
class Start:
    def __init__(self):
        self.pure_pursuit = PurePursuit()
        self.pid = PID(kp=1.0, ki=0.1, kd=0.01)

        self.local_path_sub = rospy.Subscriber('/ego_waypoint', Marker, self.local_cb)
        self.global_gps_sub = rospy.Subscriber('/current_global_waypoint', Marker, self.global_cb)
        self.pose_sub = rospy.Subscriber('/ego_pos', PoseStamped, self.pose_cb)
        self.gps_pos_sub = rospy.Subscriber('/gps_ego_pose', Point, self.gps_pos_cb)
        self.start_flag_sub = rospy.Subscriber('/start_flag', Bool, self.flag_cb)
        self.point_sub = rospy.Subscriber('/last_target_point', Marker, self.point_callback)
        self.gps_sub = rospy.Subscriber('/novatel/oem7/bestgnsspos', BESTGNSSPOS, self.bestgps_cb)
        self.yaw_sub = rospy.Subscriber('/vehicle/yaw_rate_sensor', Float32, self.yaw_cb)
        self.rl_sub = rospy.Subscriber('/vehicle/velocity_RL', Float32, self.rl_callback)
        self.rr_sub = rospy.Subscriber('/vehicle/velocity_RR', Float32, self.rr_callback)
        self.steer_sub = rospy.Subscriber('/vehicle/steering_angle', Float32, self.steer_callback)
        self.obstacle_sub = rospy.Subscriber('/mobinha/hazard_warning', Bool, self.obstacle_cb)
        self.sign_sub = rospy.Subscriber('/mobinha/is_crossroad', Bool, self.sign_cb)

        self.actuator_pub = rospy.Publisher('/target_actuator', Actuator, queue_size=10)
        self.light_pub = rospy.Publisher('/vehicle/left_signal', Float32, queue_size=10)
        self.global_odom_pub = rospy.Publisher('/global_odom_frame_point', Marker, queue_size=10)
        self.laps_complete_pub = rospy.Publisher('/laps_completed', Bool, queue_size=10)

        self.curr_v = 0
        self.pose = PoseStamped()
        self.global_waypoints_x = None
        self.global_waypoints_y = None
        self.local_waypoint = Point()
        self.steer_ratio = 12
        self.current_point = None
        self.curr_lat = None
        self.curr_lon = None
        self.current_waypoint_idx = 0

        self.global_pose_x = None
        self.global_pose_y = None
        self.yaw_rate = None
        self.is_start = False

        self.moving_average_window = 1
        self.point_history_x = deque(maxlen=self.moving_average_window)
        self.point_history_y = deque(maxlen=self.moving_average_window)

        self.rl_v = 0
        self.rr_v = 0
        self.curr_steer = 0
        self.inter_steer = 0

        self.obstacle_flag = False
        self.sign_flag = False

    def local_cb(self, msg):
        # 로컬 웨이포인트 업데이트 (필요 시 구현)
        pass

    def global_cb(self, msg):
        self.global_waypoints_x = msg.pose.position.x
        self.global_waypoints_y = msg.pose.position.y

    def pose_cb(self, msg):
        self.pose = msg
        self.yaw = self.get_yaw_from_pose(self.pose)
        self.node.x = msg.pose.position.x
        self.node.y = msg.pose.position.y
        self.node.yaw = self.yaw

    def get_yaw_from_pose(self, msg):
        orientation_q = msg.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def gps_pos_cb(self, msg):
        self.global_pose_x = msg.x
        self.global_pose_y = msg.y

    def rl_callback(self, msg):
        self.rl_v = msg.data

    def rr_callback(self, msg):
        self.rr_v = msg.data

    def flag_cb(self, msg):
        self.is_start = msg.data

    def point_callback(self, msg):
        self.current_point = Point()
        self.current_point.x = msg.pose.position.x + 2.1
        self.current_point.y = msg.pose.position.y + 0.2

    def bestgps_cb(self, msg):
        self.curr_lat = msg.lat
        self.curr_lon = msg.lon

    def yaw_cb(self, msg):
        self.yaw_rate = radians(msg.data)

    def mpc_candidate_v(nominal_v, delta=0.2, num_candidates=5):
        candidate_vs = []
        for i in range(num_candidates):
            candidate = nominal_v - delta * (num_candidates // 2) + i * delta
            candidate_vs.append(candidate)
        return candidate_vs

    def mpc_candidate_steer(self, target_point, base_steer, delta=1.0, num_candidates=5):
        candidate_steers = []
        for i in range(num_candidates):
            candidate = base_steer - delta * (num_candidates // 2) + i * delta
            candidate_steers.append(candidate)
        return candidate_steers

    def nearest_index_on_global_path(self):
        if self.global_pose_x is None or self.global_pose_y is None:
            return 0
        pos = np.array([self.global_pose_x, self.global_pose_y])
        path = np.column_stack((self.cx, self.cy))
        dists = np.linalg.norm(path - pos, axis=1)
        return int(np.argmin(dists))

    def global_to_local(self, target_point, position, yaw):
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]
        x_local = dx * cos(-yaw) - dy * sin(-yaw)
        y_local = dx * sin(-yaw) + dy * cos(-yaw)
        return x_local, y_local

    def pub_global_waypoint(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 5
        marker.scale.y = 5
        marker.scale.z = 5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        self.global_odom_pub.publish(marker)

    # 모델 예측 제어 (글로벌 웨이포인트 기준)
    def Model_Predictive_Control_Common(self, ego_ind):
        # 참조 궤적 생성 (상태: [x, y, yaw, v]) -> 여기서 sp(speed profile)는 미리 생성된 값 사용
        z_ref = calc_ref_trajectory_in_T_step(self.node, ego_ind, self.ref_path, self.sp)
        # Lookahead: 현재 인덱스에서 30 포인트 앞 target point 결정
        lookahead_offset = 30
        target_index = min(ego_ind + lookahead_offset, len(self.cx) - 1)
        target_point = (self.cx[target_index], self.cy[target_index])
        
        # Pure Pursuit을 통해 기준 조향각 산출
        base_steer, _ = self.pure_pursuit.run(self.curr_v, target_point, (self.node.x, self.node.y), self.yaw, 0)
        # PID를 통해 기준 속도 산출 (현재 위치와 target point 오차 기반)
        current_position = (self.node.x, self.node.y)
        nominal_v = self.pid.run(target_point, current_position)
        nominal_v = np.clip(nominal_v, 0, Parameter.speed_max)
        
        # 후보 생성: 각각 5개 후보 (속도와 조향각) - 여기서만 후보 입력값 생성 방식을 변경함
        candidate_vs = self.mpc_candidate_v(nominal_v, delta=0.2, num_candidates=5)
        candidate_steers = self.mpc_candidate_steer(target_point, base_steer, delta=1.0, num_candidates=5)
        
        best_cost = float('inf')
        best_pair = (None, None)
        best_candidate_index = 0
        for i in range(len(candidate_vs)):
            v_candidate = candidate_vs[i]
            steer_candidate = candidate_steers[i]
            z_bar = mpc_predict_next_state(z_ref, self.node.x, self.node.y, self.node.yaw, v_candidate, steer_candidate)
            cost = mpc_cost_function(z_ref, z_bar, candidate_steers)
            if cost < best_cost:
                best_cost = cost
                best_pair = (v_candidate, steer_candidate)
                best_candidate_index = i
        
        # 최적의 후보 쌍으로 상태 업데이트
        self.node.update(best_pair[0], math.radians(best_pair[1]), 0)
        return z_ref, best_pair[0], best_pair[1], target_point, target_index

    def run_control_loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if not self.is_start:
                rospy.loginfo("Not setting yet...")
                rate.sleep()
                continue

            # 최신 속도 계산 (후바퀴 속도의 평균 사용)
            self.curr_v = (self.rl_v + self.rr_v) / 7.2

            # 장애물 처리
            if self.obstacle_flag:
                temp = Actuator()
                temp.accel = 0
                temp.brake = np.clip(100 / max(self.curr_v, 0.1), 0, 100)
                temp.is_waypoint = 0
                rospy.logwarn("Obstacle detect, Command brake")
                self.actuator_pub.publish(temp)
                self.obstacle_flag = False
                continue

            # 신호 감지 처리
            if self.sign_flag and self.find_waypoint_section(self.curr_lat, self.curr_lon, waypoint_sections) != 6:
                temp = Actuator()
                temp.accel = 0
                temp.brake = np.clip(100 / max(self.curr_v * 3.6, 0.1), 0, 10)
                temp.is_waypoint = 0
                rospy.logwarn("Sign detect, Command slow")
                self.actuator_pub.publish(temp)
                for i in range(10):
                    rate.sleep()
                continue

            # 글로벌 웨이포인트 사용 상황
            if (self.global_pose_x is not None and self.global_pose_y is not None and 
                self.global_waypoints_x is not None and self.global_waypoints_y is not None):
                ego_ind = self.nearest_index_on_global_path()
                z_ref, v_cmd, steer_cmd, target_point, target_index = self.Model_Predictive_Control_Common(ego_ind)
                rospy.loginfo(f"Target Index: {target_index}, Target Point: {target_point}")
                rospy.loginfo(f"Control Output -> v: {v_cmd:.2f} m/s, steer: {steer_cmd:.2f} deg")
                temp = Actuator()
                temp.accel = v_cmd    # 여기서는 accel 필드에 속도 명령(v)를 넣음
                temp.steer = steer_cmd
                temp.brake = 0
                temp.is_waypoint = 0
                self.actuator_pub.publish(temp)
            # 로컬 웨이포인트 사용 상황 (글로벌 조건 미충족 시)
            else:
                if hasattr(self, "current_point") and self.current_point is not None:
                    way_x = self.current_point.x
                    way_y = self.current_point.y
                    position = (0, 0)
                    waypoint = (way_x, way_y)
                    yaw = self.yaw_rate
                    rospy.loginfo(f"current velocity: {self.curr_v}")
                    target_steering, target_position = self.pure_pursuit.run(self.curr_v, waypoint, position, 0, 0)
                    throttle = self.pid.run(target_position, position)
                    throttle *= 1.5
                    throttle = np.clip(throttle, 0, Parameter.speed_max)
                    temp = Actuator()
                    temp.accel = throttle
                    temp.steer = target_steering * self.steer_ratio
                    temp.brake = 0
                    temp.is_waypoint = 1
                    self.actuator_pub.publish(temp)
                else:
                    rate.sleep()
                    continue

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('control_node')
    start = Start()
    try:
        start.run_control_loop()
    except rospy.ROSInterruptException:
        pass
