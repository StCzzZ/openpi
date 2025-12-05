import time
import numpy as np

# Import Flexiv RDK Python library
import sys
import os
import flexivrdk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from get_ip import get_local_ip

class ModeMap:
    idle = "IDLE"
    cart_impedance_online = "NRT_CARTESIAN_MOTION_FORCE"
    joint = "NRT_JOINT_POSITION"


class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """
    logger_name = "FlexivRobot"

    def __init__(self, robot_ip_address='192.168.2.100', pc_ip_address=None, default_pose=[0.6, 0, 0.2, 0, -0.5**0.5, 0.5**0.5, 0]):
        if pc_ip_address is None:
            pc_ip_address = get_local_ip(robot_ip_address)
        
        self.robot_states = flexivrdk.RobotStates()
        # self.log = flexivrdk.Log()
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot("Rizon4-00020")
        self.default_pose = default_pose
        self.home_pose = [0.6,0,0.2,0,0,1,0]
        self.home_joint_pos = [0.21771033108234406, -0.3984721302986145, -0.03492163121700287, 1.8705854415893555, 0.0208751130849123, 0.6941397190093994, 0.1695701628923416]
        self.init_robot()
        self.init_pose = self.get_tcp_pose()
    
    def init_robot(self):
        # log = self.log
        mode = self.mode
        robot = self.robot

        # Clear fault on robot server if any
        if robot.fault():
            # log.warn("Fault occurred on robot server, trying to clear ...")
            print("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.ClearFault()
            time.sleep(2)
            # Check again
            if robot.fault():
                # log.error("Fault cannot be cleared, exiting ...")
                print("Fault cannot be cleared, exiting ...")
                return
            # log.info("Fault on robot server is cleared")
            print("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        # log.info("Enabling robot ...")
        print("Enabling robot ...")
        # robot.enable()
        robot.Enable()

        # Wait for the robot to become operational
        while not robot.operational():
            time.sleep(1)

        # log.info("Robot is now operational")
        print("Robot is now operational")

        # Move robot to home pose
        # log.info("Moving to home pose")
        print("Moving to home pose")
        # self.send_joint_pose(self.home_joint_pos)
        # time.sleep(4)
        self.send_tcp_pose(self.home_pose)
        time.sleep(4)

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        # Zero Force-torque Sensor
        # =========================================================================================
        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        robot.ExecutePrimitive("ZeroFTSensor", {}, block_until_started = False)

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        # log.warn(
        #     "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        # )
        print("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        # Wait for primitive completion
        while robot.busy():
            time.sleep(1)
        # log.info("Sensor zeroing complete")
        print("Sensor zeroing complete")

    def enable(self, max_time=10):
        """Enable robot after emergency button is released."""
        self.robot.Enable()
        tic = time.time()
        while not self.is_operational():
            if time.time() - tic > max_time:
                return "Robot enable failed"
            time.sleep(0.01)
        return

    def _get_robot_status(self):
        self.robot_states = self.robot.states()
        return self.robot_states

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.mode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.mode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.SwitchMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.

        Args:
            mode: 'idle', 'cart_impedance_online'
            sleep_time: sleep time to control mode switch time

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)
        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("[Robot] Set mode: {}".format(str(self.get_control_mode())))

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        # return self.robot.isFault()
        return self.robot.fault()

    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.stopped()

    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.connected()

    def is_operational(self):
        """Check if robot is operational."""
        return self.robot.operational()

    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_pose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_vel)

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode('cart_impedance_online')
        self.robot.setMaxContactWrench(max_wrench)

    def send_impedance_online_pose(self, tcp):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode('cart_impedance_online')
        self.robot.SendCartesianMotionForce(np.array(tcp), [0] * 6, max_linear_vel=0.1) # 0.1: maximum velocity

    def send_tcp_pose(self, tcp):
        """
        Send tcp pose.
        """
        self.send_impedance_online_pose(tcp)

    def send_joint_pose(self, q):
        """
        Send joint pose.
        """
        self.switch_mode('joint')
        DOF = len(q)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1] * DOF
        MAX_ACC = [1] * DOF
        self.robot.SendJointPosition(np.array(q), target_vel, target_acc, MAX_VEL, MAX_ACC)

    def get_robot_state(self):
        raw = self._get_robot_status()
        tcpPose = raw.tcp_pose
        tcpVel = raw.tcp_vel
        jointPose = raw.q
        jointVel = raw.dq
        return tcpPose, jointPose, tcpVel, jointVel
    
class FlexivGripper:
    def __init__(self, r: FlexivRobot) -> None:
        self.gripper_state = flexivrdk.GripperStates()
        self.gripper = flexivrdk.Gripper(r.robot)
        self.gripper.Enable("Robotiq-2F-85")
        self.gripper_state = self.gripper.states()
        print(self.gripper_state.width)
        self.max_width = 0.085
    def move(self, width):
        # self.gripper.move(self.max_width * width / 1000, 0.1, 20)
        width = np.clip(width,0,0.085)
        self.gripper.Move(width, 0.1, 25)
    def move_from_sigma(self, width):
        self.gripper.Move(np.clip(self.max_width * width / 1000,0,0.085), 0.1, 25)
    def get_gripper_state(self):
        self.gripper_state = self.gripper.states()
        return self.gripper_state.width 
    
if __name__ == "__main__":
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    while True:
        width = float(input("Enter gripper width: "))
        gripper.move(width)
        time.sleep(0.5)
        tcp_pose, joint_pos, _, _ = robot.get_robot_state()
        print(tcp_pose)
        print(joint_pos)
        import pdb; pdb.set_trace()
