Use the named ManiSkill policy cameras in the initial observation. The available tools identify which cameras provide depth or segmentation. Use the reported arm pose and the same coordinate frame for measured points and motion targets. A null table height means its height was not measured.

For two arms, each action selects left or right and both arms share the mounting midpoint. A drawing arm controls the native pen tip with move_to and has no gripper. On a mobile manipulator the arm actions leave the base and lift stationary.

When joint_control is present, drive_joints controls the listed body or finger joints. Use action_order for commands and state_order for measured positions and velocities. Each value scales that joint's max_delta_per_step, in its named state unit. Velocities use those units per second. The base pose uses the initial base origin and heading. Use small commands and observe how the body or object responds. Camera-free tasks use joint feedback.
