import math


import serial
import threading
import time
import struct
from dataclasses import dataclass, field


@dataclass
class BasicSensorData:
    bumper: int = 0
    wheel_drop: int = 0
    cliff: int = 0
    encoder_l: int = 0
    encoder_r: int = 0
    battery: int = 0
    updated_at: float = 0.0


@dataclass
class InertialSensorData:
    gyro_angle: float = 0.0
    gyro_rate: float = 0.0
    updated_at: float = 0.0


@dataclass
class OdometryState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    prev_left_ticks: int | None = None
    prev_right_ticks: int | None = None
    updated_at: float = 0.0


@dataclass
class KobukiState:
    timestamp: float = 0.0
    basic: BasicSensorData = field(default_factory=BasicSensorData)
    inertial: InertialSensorData = field(default_factory=InertialSensorData)
    odometry: OdometryState = field(default_factory=OdometryState)

class KobukiDriver:
    def __init__(
        self,
        port='/dev/kobuki',
        cmd_timeout=0.5,
        cmd_rate=10,
        wheel_separation=0.23,
        tick_to_meter=0.00008529,
    ):
        """
        Initialize Kobuki driver with command handling.
        
        Args:
            port: Serial port (default '/dev/kobuki')
            cmd_timeout: Command timeout in seconds (default 0.5s)
            cmd_rate: Command send rate in Hz (default 10 Hz)
            wheel_separation: Distance between wheels in meters (default 0.23)
            tick_to_meter: Encoder tick to meter conversion factor
        """
        # Configuration parameters (can be updated by ROS later)
        self.cmd_timeout = cmd_timeout
        self.cmd_rate = cmd_rate
        self.cmd_period = 1.0 / cmd_rate

        self.ser = None
        self.running = False
        self._state_lock = threading.Lock()
        self._serial_lock = threading.Lock()
        self._cmd_lock = threading.Lock()

        # Connection Settings
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.1)
        except Exception as e:
            print(f"Error connecting to Kobuki: {e}")
            return

        # Robot Physical Constants (for Odom)
        self.TICK_TO_METER = tick_to_meter
        self.WHEEL_BASE = wheel_separation

        # Structured state storage.
        self.state = KobukiState()

        # Command storage (latest velocity command)
        self._cmd_linear = 0.0  # mm/s
        self._cmd_angular = 0.0  # rad/s
        self._cmd_timestamp = time.time()

        # Start Background Threads
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        self.cmd_thread = threading.Thread(target=self._cmd_loop)
        self.cmd_thread.daemon = True
        self.cmd_thread.start()

    def _run(self):
        """State machine to parse incoming serial bytes."""
        while self.running and self.ser is not None:
            # 1. Look for Header [0xAA, 0x55]
            if self.ser.read(1) == b'\xAA':
                if self.ser.read(1) == b'\x55':
                    # 2. Read Length
                    len_byte = self.ser.read(1)
                    if not len_byte: continue
                    length = ord(len_byte)
                    
                    # 3. Read Payload and Checksum
                    payload = self.ser.read(length)
                    checksum_byte = self.ser.read(1)
                    if not checksum_byte: continue
                    checksum = ord(checksum_byte)

                    # 4. Verify Checksum (XOR of Length + Payload)
                    calculated_cs = length
                    for b in payload:
                        calculated_cs ^= b
                    
                    if calculated_cs == checksum:
                        self._parse_payload(payload)

    def _parse_payload(self, payload):
        """Splits the payload into sub-packets based on ID."""
        i = 0
        packet_ts = time.time()
        with self._state_lock:
            self.state.timestamp = packet_ts
        
        while i < len(payload):
            sub_id = payload[i]
            sub_len = payload[i+1]
            sub_data = payload[i+2 : i+2+sub_len]

            if sub_id == 0x01: # Basic Sensor Data
                # Unpack 15 bytes of basic data
                # Bumper(1), WheelDrop(1), Cliff(1), L_Enc(2), R_Enc(2)...
                b, wd, c, l_enc, r_enc = struct.unpack('<BBBHH', sub_data[0:7])
                with self._state_lock:
                    self.state.basic.bumper = b
                    self.state.basic.wheel_drop = wd
                    self.state.basic.cliff = c
                    self.state.basic.encoder_l = l_enc
                    self.state.basic.encoder_r = r_enc
                    self.state.basic.battery = sub_data[11]
                    self.state.basic.updated_at = packet_ts
                    self._update_odom(l_enc, r_enc, packet_ts)


            elif sub_id == 0x0D: # 3D Inertial Sensor (Gyro)
                # Unpack 3 signed shorts (6 bytes) for X, Y, and Z raw data
                raw_x, raw_y, raw_z = struct.unpack('<hhh', sub_data[2:8])

                # Kobuki conversion constant: 0.00875 deg/s per digit
                digit_to_dps = 0.00875

                with self._state_lock:
                    # Apply the 90-degree CCW Z-axis rotation matrix mapping
                    self.state.inertial.gyro_rate =  digit_to_dps * raw_z


            i += (2 + sub_len)

    def _update_odom(self, l_ticks, r_ticks, update_ts):
        """Calculates X, Y, Theta and handles 16-bit encoder wrap-around."""
        odom = self.state.odometry

        if odom.prev_left_ticks is None:
            odom.prev_left_ticks = l_ticks
            odom.prev_right_ticks = r_ticks
            odom.updated_at = update_ts
            odom.linear_velocity = 0.0
            odom.angular_velocity = 0.0
            return

        def handle_wrap(curr, prev):
            diff = curr - prev
            if diff > 32768: diff -= 65536
            elif diff < -32768: diff += 65536
            return diff

        dl = handle_wrap(l_ticks, odom.prev_left_ticks) * self.TICK_TO_METER
        dr = handle_wrap(r_ticks, odom.prev_right_ticks) * self.TICK_TO_METER
        dt = update_ts - odom.updated_at if odom.updated_at > 0.0 else 0.0
        
        # Differential-drive odometry:
        # d = (dl + dr) / 2, dtheta = (dr - dl) / wheel_base
        d = (dl + dr) / 2.0
        dtheta = (dr - dl) / self.WHEEL_BASE
        
        odom.x += d * math.cos(odom.theta)
        odom.y += d * math.sin(odom.theta)
        odom.theta += dtheta
        if dt > 0.0:
            odom.linear_velocity = d / dt
            odom.angular_velocity = dtheta / dt
        else:
            odom.linear_velocity = 0.0
            odom.angular_velocity = 0.0
        odom.updated_at = update_ts
        
        odom.prev_left_ticks = l_ticks
        odom.prev_right_ticks = r_ticks

    def set_velocity(self, linear_vel, angular_vel):
        """
        Store the latest velocity command (thread-safe).
        Command will be sent periodically by the background command loop.
        
        Args:
            linear_vel: Linear velocity in mm/s
            angular_vel: Angular velocity in rad/s
        """
        with self._cmd_lock:
            self._cmd_linear = linear_vel
            self._cmd_angular = angular_vel
            self._cmd_timestamp = time.time()

    def _send_drive_cmd(self, linear_vel, angular_vel):
        """
        Internal method to send drive command to robot.
        linear_vel: mm/s
        angular_vel: rad/s
        """
        # Kobuki expects Speed (mm/s) and Radius (mm)
        # Radius = v / w
        if abs(angular_vel) < 0.001:
            radius = 0
        elif abs(linear_vel) < 10:
            radius = 1 if angular_vel > 0 else -1
            linear_vel = float((angular_vel * 1000 * self.WHEEL_BASE) / 2)
        else:
            radius = int(linear_vel / angular_vel)
            # Clip radius to Kobuki limits
            if radius > 2500: radius = 0
            elif radius < -2500: radius = 0

        # Build Packet
        payload = bytearray([0x01, 0x04])
        payload += int(linear_vel).to_bytes(2, 'little', signed=True)
        payload += int(radius).to_bytes(2, 'little', signed=True)
        
        header = bytearray([0xAA, 0x55, len(payload)])
        packet = header + payload
        
        cs = len(payload)
        for b in payload: cs ^= b
        packet.append(cs)
        
        if self.ser is None:
            return

        with self._serial_lock:
            self.ser.write(packet)

    def _cmd_loop(self):
        """
        Background thread that sends commands at cmd_rate (10 Hz default).
        Implements timeout: stops robot if no new command in cmd_timeout (0.5s default).
        """
        last_send = time.time()
        
        while self.running:
            now = time.time()
            
            # Read current command with lock
            with self._cmd_lock:
                cmd_linear = self._cmd_linear
                cmd_angular = self._cmd_angular
                cmd_age = now - self._cmd_timestamp
            
            # Check timeout: if command is stale, send stop command
            if cmd_age > self.cmd_timeout:
                self._send_drive_cmd(0, 0)
            else:
                self._send_drive_cmd(cmd_linear, cmd_angular)
            
            # Sleep to maintain cmd_rate
            last_send = now
            time.sleep(self.cmd_period)

    def drive(self, linear_vel, angular_vel):
        """
        Deprecated: Use set_velocity() instead.
        Kept for backwards compatibility.
        """
        self.set_velocity(linear_vel, angular_vel)

    def get_state(self):
        """Returns a thread-safe copy of the latest flattened state."""
        with self._state_lock:
            return {
                "timestamp": self.state.timestamp,
                "bumper": self.state.basic.bumper,
                "wheel_drop": self.state.basic.wheel_drop,
                "cliff": self.state.basic.cliff,
                "encoder_l": self.state.basic.encoder_l,
                "encoder_r": self.state.basic.encoder_r,
                "battery": self.state.basic.battery,
                "gyro_angle": self.state.inertial.gyro_angle,
                "gyro_rate": self.state.inertial.gyro_rate,
                "x": self.state.odometry.x,
                "y": self.state.odometry.y,
                "theta": self.state.odometry.theta,
                "linear_velocity": self.state.odometry.linear_velocity,
                "angular_velocity": self.state.odometry.angular_velocity,
                "basic_timestamp": self.state.basic.updated_at,
                "inertial_timestamp": self.state.inertial.updated_at,
                "odometry_timestamp": self.state.odometry.updated_at,
            }

# --- Main Test Loop ---
if __name__ == "__main__":
    # Create driver with custom command timeout and rate (optional)
    robot = KobukiDriver(
        port='/dev/kobuki',  # Adjust as needed
        cmd_timeout=0.5,   # 0.5s timeout before stopping
        cmd_rate=10        # Send commands at 10 Hz
    )
    
    try:
        print("Robot started. Press Ctrl+C to stop.")
        print(f"Command rate: {robot.cmd_rate} Hz, Timeout: {robot.cmd_timeout}s")
        
        while True:
            # Set velocity command (will be sent continuously by background thread)
            robot.set_velocity(100, 0.5)  # Drive in a slow circle
            
            data = robot.get_state()
            
            time.sleep(0.2)  # Update display at 5 Hz
    except KeyboardInterrupt:
        # Stop robot
        robot.set_velocity(0, 0)
        time.sleep(0.1)  # Allow time for stop command to be sent
        robot.running = False
        print("\nStopped.")
