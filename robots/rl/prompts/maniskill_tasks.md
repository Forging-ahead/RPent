# ManiSkill task instructions

## DrawSVG-v1

Trace the actual black SVG on the white canvas, including its interior turns. If the pen hides the outline, move
it up and aside first. Pick two consecutive visible corners, back-project their pixels, then immediately execute
hover, lower to canvas height, and draw that first segment. Continue from its endpoint in small batches. Lift
across visible gaps. Re-image after each batch instead of mapping the entire outline upfront.

## DrawTriangle-v1

Move the pen up and clear of the outline. Use that unobstructed base view for all three corners. This triangle
is equilateral. Back-project the two clearest black centerline junctions. Check the third against the
equilateral apex on the visible side and verify all three side lengths agree within about 1 cm before touching.
Hover over the verified corner, lower vertically, and follow each edge through its midpoint to the next corner
at canvas height.

## FMBAssembly1Easy-v1

Start by uncovering the loose GREEN bridge: shift the hand one palm-width away from the loose parts while
keeping it high, xyz only. Preserve that measured wrist through the first grasp and lift; rotate only when
carrying the green bridge. In the clear base-camera view find two green feet joined by a green rail.
Back-project pixels INSIDE the green rail; check the planned grasp lies on that same green strip, away from the
isolated black plate and red fixture. Grip the narrow solid rail midway between feet, from above, about 1.5 cm
below its top. Lift, checking green feet move, turn the bridge upright, set feet on clear nearby table, and
regrasp the top rail from above. Align the feet over the two mounting slots around the yellow peg, lower and
release.

## InsertFlower-v1

Pinch the thicker neck where the white bloom meets the stem. Back-project the neck, its height and a clear patch
on its side. Set the required pinch wrist RPY [-1.67,-0.05,-1.57] during the approach to the clear side patch,
then retain it. At opening 0.50, keep the thumb/index gap OPEN until the neck is between the pads. At closure
the pinch center is roughly palm+[10.3,2.8,-6.2] cm. Lower beside the neck on bare table first, keeping
clearance under the pads; slide in horizontally at neck height, then close once to 0.32. If contact moves the
flower, remeasure its new neck before gripping. Watch the flower move with the first carry. Rotate stem-end down
over the mouth, insert and release.

## LiftPegUpright-v1

Measure the red-blue seam at the middle of the 24 cm peg. In one execute_actions issue exactly three actions:
(1) move_to the seam XY at table+2.5 cm, omit orientation; (2) close fully; (3) tilt and lift the gripped
midpoint simultaneously, moving about one half-length toward the pedestal to table+12 cm. Keep gripping. For a
shaft along robot X with the starting top-down wrist yaw near -pi/2, retain yaw, pitch 0, and command roll +1.30
radians, slightly through vertical, so the peg sweeps through upright. The rising midpoint gives the lower end
clearance in this combined move.

## PegInsertionSide-v1

Before the first move, locate hole center, inward normal, and orange tip. Grasp the white shaft 2–3 cm behind
its color seam. For a top-down parallel gripper the grasp line is UNORIENTED: compare candidate yaw and yaw+pi
to measured wrist yaw; wrap both differences to [-pi,pi] and pick the SMALLER absolute turn (at most pi/2). Keep
measured roll/pitch. Descend, close and lift. Compute actual grip-to-tip vector including the orange section.
Rotate the held tip by the signed turn from shaft to inward normal. At that RPY queue the pre-insertion pose 2
cm outside, then advance into the measured hole along its normal; use the same RPY throughout insertion.

## PickCube-v1

The wrist starts pointing down. Omit orientation_rpy from move_to to keep that orientation. Use the segmented
cube's x and y. For the grasp height, add half of cube_size_cm to table_height_above_base_cm. The initial wrist
pose is already a clear hover. Move directly to the grasp position, then close. Once is_grasped is true, carry
the cube centre directly to goal.xyz and hold it still with the gripper closed.

## PickCubeSO100-v1

The SO100 needs four motions. Measure the cube and green sphere before moving. Preshape in air to opening 0.25.
Move directly from there to the cube center XY at table + half the cube height + 0.1 cm, XYZ only; the diagonal
descent is the approach. Grip at 0.12. In one diagonal carry aim the cube at the exact CENTER of the goal. For
the sphere use the midpoint of opposite contour extrema in table XY, top minus radius in Z; add the closed
TCP-minus-cube-center offset for the TCP target. Keep holding. If time remains and the cube is outside, use its
measured residual as a translation from the actual TCP.

## PickCubeWidowXAI-v1

The measured reset wrist and jaw opening already suit this cube. Keep both for approach. The first motion is
XYZ-only, without a preparatory rotation. Localize the cube by opposite visible footprint edges and the goal by
silhouette center and sphere top minus radius. Descend in one XYZ-only move to the cube footprint center at
mid-height; close FULLY to 0 and keep 0. A blocked closure goes straight to the goal on the next action. Aim the
cube about half a goal radius inward from sphere center toward the robot base (horizontal only), leaving
clearance inside the sphere; correct the target by the measured closed TCP-minus-cube offset. Carry with the
pitch/yaw measured AFTER closure and roll about 0.2 rad below the grasp roll. Omit a test lift. Hold the closed
grip once in range to let the arm settle.

## PickSingleYCB-v1

For an elongated YCB item, measure the closer short end relative to the starting TCP. Grasp one pad inside this
end at its UPPER rim, with the jaw pads overlapping the top part of both side walls; minimize downward travel.
If the open jaws already span the object, keep the reset wrist and descend XYZ only. If rotation is necessary
from the image, use the nearest equivalent jaw yaw in the descent command itself. Close fully. One continuous
carry now goes to the lower interior of the sphere. Tilt the long free end upward only about 30 degrees while
translating; large wrist turns slow the lift. Compute TCP position from the desired object center and the
rotated grip offset. For compact objects grip the upper sides centrally and keep grasp orientation. Keep closed
at the goal.

## PlaceSphere-v1

Budget 50 steps: grasp by about 23, one carry of roughly 14–17 steps, begin releasing by about 40. The first
motion goes straight to the sphere equator using XY from its top depth band, then close fully. Following contact
send [move_to(bin center XY, rim top + ball radius + 0.3 cm), set_gripper(1)] in one call. Start this low
diagonal carry directly from the grasp. It both lifts and transfers the ball. A centered low release lets it
settle in the bin.

## PlugCharger-v1

Measure charger body by the full pair of its long edges before grasp. Approach with measured wrist; include a
needed jaw yaw in the approach XYZ. At the subsequent staging pose after grasp use the image of the held charger
to check the pin axis against the socket normal; choose the smallest yaw correction aligning the axes. Socket
opening is 5 cm below top on its front wall. Pin root is 1.2 cm below the plug top on its forward face. Compute
TCP from the measured pin-root offset. Stage 2 cm before the front wall and advance along the hole normal. If
blocked, keep the plug gripped; back out 1 cm along the insertion axis, offset laterally or vertically by 0.15
cm, and reinsert. Alternate signs on subsequent attempts. Stay on this approach line and use the existing camera
views.

## PokeCube-v1

Grip the peg with its far end exposed and record TCP z after closing. Keep precisely that measured z for BOTH
moves: shift the tip onto the cube-to-bullseye line immediately behind the cube, then push far enough along that
line for the cube center to enter the bullseye. Derive each TCP target from the current tip offset. Keep wrist
and grip fixed.

## PullCube-v1

Keep the measured downward wrist orientation throughout the pickup and pull. Descend to the visible cube, close
and lift slightly. Pull toward the goal along the cube-to-goal line without a separate wrist turn.

## PullCubeTool-v1

Before grasp measure both ends of the hook blade, the blue cube footprint, and direction toward the base.
Approach low above the straight handle with the measured wrist. Descend vertically and close. A needed jaw
alignment from the image belongs in the approach XYZ command. With the blade-to-TCP offset, precompute THREE
translations and issue them as ONE execute_actions sequence holding the wrist: (1) one oblique slide around the
nearest free side, ending just past the far face with the blade outside the cube width; (2) shift sideways at
that far depth so the blade overlaps the back face; (3) pull 18 cm toward the base. Keep the blade at
cube-midheight in all three targets. Use the first slide to lift to working height as it moves; no separate
lift, exploratory partial advances or inspection stop.

## PushCube-v1

Close the jaws and omit orientation_rpy for approach and push. Reach contact height just behind the cube on the
cube-to-goal line, then push along that line at the same height.

## PushT-v1

Measure red and gray T orientations from the uninterrupted 20 cm rear crossbar edges, using the stem to resolve
sign. Choose the rotation sign on the table. For each crossbar end imagine a tangential push with that turn
sign; prefer the end whose push ALSO translates red toward gray. Measure the red contact and the gray target
before moving. Using xyz only (the pusher wrist stays as measured), queue a direct diagonal move to bare table
just outside the chosen end at half block height, then a 10–12 cm sweep through that end. Read the exposed red
edges from the end frame. Continue at contact height with a second turning or translating push to align the T; a
whole-arm retreat consumes the remaining stroke time.

## PutCarrotOnPlateInScene-v1

For this fixed 512px Bridge camera use the visible [row,col]: r=(row-256)/100, c=(col-256)/100, d=1+.158*r.
Table-object XY in mount cm is about [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Measure the thick orange
body and plate first. While still at the initial TCP HEIGHT, move above the thick body and finish the crosswise
wrist turn. Only then descend straight to z≈2 cm with that wrist fixed and close. Once the close is blocked on
the carrot, carry diagonally straight to the plate center at z≈4–5 cm and open there. Skip the separate vertical
test lift.

## PutEggplantInBasketScene-v1

In the fixed 512px sink view take the eggplant body's [row,col]. With r=(row-256)/100, c=(col-256)/100,
d=1+.157*r, approximate its XY at z=8 cm by [(29.2+3.68*c-4.25*r)/d,(-7.75-8.2*c-5.22*r)/d]. Mark the center of
the yellow basket floor in the FIRST view. Apply the same formula to that pixel. Convert its output (x,y) to the
higher basket plane as (-.3+.774*(x+.3), -21+.774*(y+21)). After gripping and lifting clear of the rim, batch:
move over that basket XY, lower there to z≈18 cm, open. Use the recorded basket position even when the carried
eggplant obscures it.

## PutSpoonOnTableClothInScene-v1

For this fixed 512px Bridge camera use the visible [row,col]: r=(row-256)/100, c=(col-256)/100, d=1+.158*r.
Table-object XY in mount cm is about [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Locate green handle midpoint
and towel center in the opening view. Batch a direct move to the handle at z≈2 cm and close immediately. If
needed for a crosswise grip, set the closest equivalent jaw yaw on that first move. At handle height avoid a
second downward probe. On contact, carry in one diagonal to towel center at z≈3–4 cm and open there.

## RollBall-v1

This is a single paddle strike. Measure blue ball center B, its radius r, and unit table direction u toward red.
Pick the low launch point B-(r+2 cm)*u on clear table behind the ball. The diagonal approach ends on bare table
clear of the ball. Send the WHOLE swing as one execute_actions list in this order: set_gripper(0.25),
move_to(launch XY, table+r), move_to(B+24 cm*u, table+r+4 cm). Use measured wrist, xyz only. The first move is
the approach, the second is the follow-through. Let the ball roll independently after the hit; use a return move
to give it simulation time.

## SO100GraspCube-v1

Save the start pose and pre-close to 0.25. For cube-center pixel (col,row), U=2*col/W-1, V=2*row/H-1,
t=33.7/(0.570+0.401*V). The table XY estimate F(pixel) in cm is [30+t*(-0.684+0.271*U+0.231*V),
-50+t*(0.456+0.406*U-0.154*V)]. Approach there with the measured wrist. Near cube height compare the gap
midpoint pixel with the cube center; add F(cube)-F(gap) to measured XY. Descend to about 1.4 cm, gently close to
0.12 and lift toward the saved start with this grip. Reopen above the table before any retry.

## StackCube-v1

The wrist starts pointing down in a clear hover. Keep that orientation and move directly to the red cube centre,
half its height above the table. Close, then plan a single carry that clears the green cube before the cubes
overlap. Align the red cube centre over the green top, accounting for any offset from the gripper, and open
immediately with time left for settling. For gaps over 15 cm, budget release before carrying. On a clear path,
batch one diagonal move_to to the release pose followed by set_gripper with opening 1. The diagonal itself
raises the cube during transfer. The full sequence has {max_episode_steps} environment steps. Use the red cube
center for the grasp, about 0.1 cm above its mid-height. For a long carry, offset the release center from green
toward red on each axis separately, by up to 1 cm in x and 1.5 cm in y, while keeping the red center over green.
Set the release height to green's top plus half the red cube height and 1.5 cm of clearance. Batch this single
diagonal carry and full opening. Reach the release pose by step 48.

## StackGreenCubeOnYellowCubeBakedTexInScene-v1

For this fixed 512px Bridge camera use the visible [row,col]: r=(row-256)/100, c=(col-256)/100, d=1+.158*r.
Table-object XY in mount cm is about [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Use centers of the green and
yellow cube boxes. From a low hover over green, lower the TCP to the middle of the 3cm cube and close in one
batch. Lift clear, carry over yellow, lower green onto its top and open.

## StackPyramid-v1

Choose the red and green base positions before moving. When one base cube is already stable, place the other
beside it and leave the pair in place. Pick up blue only after verifying the base, then release it over the
midpoint at the measured height of the two cube tops.

## TwoRobotPickCube-v1

Measure the red cube width w. Preserve each arm’s measured wrist by omitting orientation on all translation
moves. Left directly grasps cube center, then carries it low to a point HALF a cube width past the midline
toward RIGHT, at table+w. Release to opening 0.75 and clear left by one cube width diagonally up toward its own
side. Remeasure the cube. Right goes straight to the center of its upper half (table+0.7w) and closes in place
when the fingers straddle it. A stalled descent within 1 cm with the cube inside the jaw gap is a grasp
opportunity. Use that frame to locate the now-visible green sphere. Carry straight there, keeping the measured
cube-center offset in the right grasp.

## TwoRobotStackCube-v1

Green is the BASE and can slide on the table. Measure blue and the target ring. Pick a landing point just inside
the ring on the GREEN side, at least one cube width clear of blue. Keep each measured wrist throughout, using
XYZ alone. Right makes a flat pusher: close fingers in free space and position them on the table side of green
FARTHEST from the landing point, at mid-block height. Use the side view or newly unoccluded green top to center
the contact. Sweep horizontally through green toward that landing point in one stroke, stopping the finger
contact face half a cube width short of it. Lift right clear. Measure the actual green top. Left takes blue
straight at center height, closes, carries to one cube height directly above green center, and opens.

## UnitreeG1PlaceAppleInBowl-v1

Use right hand. Measure the fruit equator and bowl rim. First move directly to grasp level: apple center biased
4 cm toward robot left and 1 cm toward torso, RPY [1.9,-0.15,2.02]. If the fingers surround the fruit, close
fully even if that reach reports stalled. Keeping the MEASURED closure wrist, lift just one fruit radius in
place and inspect actual apple height in the head view. If it rose with the hand, keep that hold and transport
low toward the near edge of the bowl. Change wrist only as needed to clear the rim. Carry above the bowl center,
release, then raise the hand at least 13 cm above the bowl bottom. If fruit stayed down, reopen and reposition
once at its newly measured center. Apply any necessary wrist change in the same move as reaching its contact
position.

## UnitreeG1TransportBox-v1

Use the entire visible top rectangle (36 by 24 cm); its center is the midpoint of opposite corners. Define u
along the long axis pointing toward robot forward, v perpendicular toward robot left. Right palm approaches the
NEGATIVE long-axis end, left palm the POSITIVE end. Both contact patches are about 7 cm along +v from their
respective end centers and 7 cm below the top. Start with the nearer right hand; go directly beside its end with
the open palm facing the box, then engage left at the opposite end. Keep the palms open to squeeze the end
faces; a free-space fist wastes the grasp. Verify that both palms touch before lifting. Lift just clear of the
table, carry toward the other surface maintaining pressure, set down and release. Apply any necessary wrist
change in the same move as reaching its contact position.
