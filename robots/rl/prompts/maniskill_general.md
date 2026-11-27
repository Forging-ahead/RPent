Use the named ManiSkill policy cameras in the initial observation. The available tools identify which cameras provide depth or segmentation. Use the reported arm pose and the same coordinate frame for measured points and motion targets. A null table height means its height was not measured.

For two arms, each action selects left or right and both arms share the mounting midpoint. A drawing arm controls the native pen tip with move_to and has no gripper. On a mobile manipulator the arm actions leave the base and lift stationary.

During an approach, preserve the selected arm's measured wrist orientation by omitting orientation_rpy from move_to. Change it when the next grasp or tool contact requires a different orientation.

A clear initial TCP can serve as the approach hover. Use only as much clearance as the next move needs, and reserve steps for placement or the final goal. If an intermediate move stalls, check the measured TCP and the remaining steps. Continue from the measured pose when it already clears the next move.
