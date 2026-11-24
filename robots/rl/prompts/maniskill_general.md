Control the robot in the reported ManiSkill task. Use the initial observation to identify the task, the active controllers, and the order and bounds of the native action vector. Each vector in execute_actions advances one environment control step. Send one tool call per reply. Inspect the measured result before choosing the next actions.

Use the available policy observations. When view_env is available, choose a named camera and keep its step with any pixel you query. Use back_project for measurements on cameras with depth. Receive images through view_env. In tasks without cameras, use the reported state to choose controls.

Use the state returned after each action call to choose your next controls. The episode ends when the environment reports termination or truncation.
