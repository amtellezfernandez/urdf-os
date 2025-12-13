import logging
import math

logger = logging.getLogger(__name__)


# NOTE: This only works for the SO101 Follower robot (for now)
def get_joint_positions_from_robot(
    joint_names: list[str], robot
) -> dict[str, float] | None:
    """
    Get the current joint positions from the robot

    Args:
        robot: The robot instance (SO101Follower)

    Returns:
        Dictionary mapping joint names (from frontend) to radian values
    """
    try:
        # Get the current observation from the robot
        observation = robot.get_observation()

        # Ordered list of robot motors in SO101Follower. We pair these with the
        # provided joint_names list from the frontend BY INDEX.
        # ASSUMPTION: joint_names[i] corresponds to motor_order[i].
        # If the lengths differ, we warn and fall back to defaults for missing names.
        motor_order = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        # If the caller provides names and they don't match count, warn once.
        if len(joint_names) != 0 and len(joint_names) != len(motor_order):
            logger.warning(
                "joint_names length (%d) does not match motor count (%d). "
                "Using provided names by index where possible; defaulting to MJCF names for others.",
                len(joint_names),
                len(motor_order),
            )

        # Build motor->targetJointName map using provided names or defaults
        motor_to_joint_name: dict[str, str] = {}
        for idx, motor_name in enumerate(motor_order):
            if idx < len(joint_names) and joint_names[idx]:
                motor_to_joint_name[motor_name] = joint_names[idx]
            else:
                # Default to MJCF motor (joint) name, e.g., "shoulder_pan"
                motor_to_joint_name[motor_name] = motor_name

        joint_positions: dict[str, float] = {}

        # Extract joint positions and convert degrees to radians
        for motor_name, target_joint_name in motor_to_joint_name.items():
            motor_key = f"{motor_name}.pos"
            if motor_key in observation:
                angle_degrees = observation[motor_key]
                angle_radians = angle_degrees * (math.pi / 180.0)
                joint_positions[target_joint_name] = angle_radians
            else:
                logger.warning("Motor %s not found in observation", motor_key)
                joint_positions[target_joint_name] = 0.0

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return None
