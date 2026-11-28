# ManiSkill task instructions

## DrawSVG-v1

Look once with the base camera. If the pen hides the outline, make the first move a vertical lift of about
12 cm from its measured pose, keeping x, y and wrist orientation. Keep the clear view as your map and segment
the black path once. Back-project both centerline endpoints of one visible edge. As soon as you have them,
batch a hover over the first, a descent to its measured canvas height, and a stroke to the second. Continue
edge by edge, reusing the last endpoint. Lift across gaps and check the ink against the map for missing strokes.

## DrawTriangle-v1

Look at the triangle with the base camera. If the pen hides a corner, lift it over a clear margin and take a
fresh view. Keep that frame as your map. Back-project the two clearest corners on the black centerline.
Once you have one edge, hover above its first corner, lower to the measured canvas height and trace the edge
through its midpoint. Recover the third corner from the map, using the equilateral apex on the visible side
as a check. Trace the remaining two edges through their midpoints and close the triangle at the first corner.

## FMBAssembly1Easy-v1

Identify one exposed solid green FOOT with free space on both sides; point-project the top of that foot.
Center the open jaws across its short width from above. If necessary turn yaw crosswise on the approach,
then descend to about 1 cm below its top and close fully. Lift and rotate the held bridge upright. Before
the hand covers the board, save the midpoint of the yellow peg between the two colored supports. After
lifting, measure the carried green center relative to the TCP once. Batch the transfer above the saved
midpoint and the descent along it: the seated bridge center is about 4.9 cm above the support tabletop.
Carry the measured TCP offset through both targets. Open after seating.

## InsertFlower-v1

Move the hand sideways once to expose the flower. Save vase-mouth center and height BEFORE grasping. Use a
low side pinch on the bloom base where it joins the stem. At RPY [-1.67,-0.05,-1.57] position the open palm
behind the neck, slide the gap beneath the bloom until pads surround it; a partial move at the table can
already be in contact. Close to 0.05 immediately. While holding, use ONE diagonal transfer upward and toward
the saved mouth rather than a separate high lift. Track whether the bloom follows; once over the mouth guide
the stem downward and open. If the first pinch slipped, reopen and use a lower contact at the base of the
bloom.

## LiftPegUpright-v1

Pinch the rod near its end rather than at the seam, about 8 cm from the midpoint toward the base when the
shaft lies along robot X. Use exactly one three-item execute_actions batch: (1) directly descend with
measured wrist to the selected point at table+2.5 cm; (2) close fully; (3) keeping that grip, raise this end
and roll simultaneously toward vertical. For a top-down grasp with yaw near -pi/2 use final roll +1.1, pitch
0 and that yaw. The grasp is 8 cm ABOVE the midpoint when upright, so finish at table+20 cm and translate
toward the other end during the rise. Start with the actual reset hover.

## PegInsertionSide-v1

Grasp the WHITE shaft 2–3 cm behind the color seam. Observe the orange TIP and the hole rim in the initial
hand view and save their world measurements. Use a crosswise top-down grasp. After closing, carry the held
tip in one diagonal to a staging point 2 cm outside the hole; set the alignment yaw IN THIS SAME
translation, preserving measured roll and pitch. Use the feature vector from grip to orange TIP to offset
the TCP. Advance along the inward normal. If rim contact stalls, withdraw 0.5 cm and raise a low tip or
center a laterally displaced tip using the rim in the wrist view; attempt insertion again with the same
axial direction.

## PickCube-v1

The wrist starts pointing down. Omit orientation_rpy from move_to to keep that orientation. Use the segmented
cube's x and y. For the grasp height, add half of cube_size_cm to table_height_above_base_cm. The initial wrist
pose is already a clear hover. Move directly to the grasp position, then close. Once is_grasped is true, carry
the cube centre directly to goal.xyz and hold it still with the gripper closed.

## PickCubeSO100-v1

Save the green sphere center using its contour and top minus radius. First batch: set_gripper(0.25), move
directly to cube XY at table+1.35 cm keeping wrist, set_gripper(0.12). Next action is the FULL diagonal
carry to the sphere center plus measured TCP-minus-cube offset. For the rise with a small sideways shift,
keep measured pitch; use absolute carry roll = measured closing roll +0.15 rad and yaw = measured closing
yaw +0.10 rad for a carry from negative base Y toward decreasing base X (wrap at pi). This allows the wrist
to turn as the shoulder pans. Use the goal for this first lift and maintain the closed grip at arrival.

## PickCubeWidowXAI-v1

Measure the red cube and green sphere in the reset view. The open jaws already clear the table. Use three
physical commands: (1) move straight from reset to the cube mask's reported XY at table + 1.9 cm, XYZ only;
(2) close to 0.30; (3) carry diagonally to the sphere in one move. The first target is the grasp itself,
and the carry is the lift. A partial close with a measured gap above 0.30 is expected on contact. Use the
measured sphere center, biased 0.5 cm toward the base, plus the measured TCP-minus-cube offset at grasp.
Set carry roll to the measured closing roll minus 0.20 rad, keeping pitch and yaw. Keep holding at arrival.

## PickSingleYCB-v1

Pick with the smallest lowering distance. For the block, keep reset wrist and select the upper sidewalls
near the short end closest to the TCP. The finger pads should overlap just 0.5–1 cm of the upper sidewall
with the TCP at the measured top height. Batch direct descent and closure. Read the green marker in the
CURRENT base view after closure; point-segment or back-project its visible center. The carry goes to the
lower interior of the goal sphere in one diagonal; rotate the projecting far end upward during that carry so
the object center rises relative to the grasp. Keep holding.

## PlaceSphere-v1

Segment “blue sphere” and “white shallow bin tray”. Use the top 1 cm of the sphere mask for grasp XY, and
the whole tray mask for bin center XY. Keep measurements to 0.1 cm; the sphere must settle within 0.5 cm
of the center. With the starting wrist, move directly to the sphere equator and close. After contact,
batch a diagonal move to bin center at rim top + sphere radius + 0.3 cm with full opening.

## PlugCharger-v1

First pick the small loose white body with its two metal pins: measure its rectangular top, retain the
measured downward wrist, descend to body mid-height and close to 0.25. Finish this pickup BEFORE inspecting
the outlet. The receptacle front is a 10 cm square; hole midpoint is 5 cm below its top. Keep the plug
level, align its pins with the measured normal, and approach with the pin-root midpoint aimed there. The
pins are 1.6 cm long: when tips just touch, the body is still 1.6 cm short. If blocked, withdraw, align the
two holes' midpoint from their symmetric positions to submillimeter precision, then advance the full pin
length. Keep each correction short.

## PokeCube-v1

From the base view measure the shaft midline, working tip, cube and target. Grasp about 1 cm from the shaft
midpoint toward its blunt end. First batch: hover 4 cm above the shaft top; descend at the same XY straight to
1.8 cm BELOW the top; close fully. These are the only approach waypoints. At finger contact use the
just-measured TCP z for both remaining moves. Batch a lateral alignment that brings the tip to the cube's near
face on the push line, then a push along that line into the circle. This alignment also tests the grasp. Keep
the grip and wrist still to settle.

## PullCube-v1

Measure the blue cube and bullseye in the base view. Keep the reset wrist and grasp at the cube mask's XY,
about two-thirds of the cube height above the table, then close fully. Lift to table + 8 cm. From this
lifted pose, pull diagonally toward the bullseye in the very next move, ending 2 cm inside the near edge
of its 10 cm radius at table + 5.5 cm. Keep holding. Success is measured in XY as the cube enters the
circle, so a separate descent to the table or release spends steps needed for the pull.

## PullCubeTool-v1

Grip the handle 2 cm from its free end at 3 cm above the tabletop, using the reset wrist. Close. Use the low
L-shaped hook without a test lift. The next four TCP targets all have z = table + 3.5 cm. In the first
batch, (1) move sideways 8 cm AWAY from cube (if handle y is smaller than cube y, subtract 8 cm); (2) move
forward until blade is just past the cube; (3) move sideways toward cube until the inner edge overlaps its
far face. Next action is one straight pull 18–22 cm toward the base, fixed y and z. The blade travels at
cube mid-height on all four strokes. Measure before grasp so each slide can follow immediately.

## PushCube-v1

Segment “blue cube” and “red and white goal target”. Close the jaws and keep the measured wrist. Approach
5 cm behind the cube on the cube-to-goal line at cube top + 5 cm. Descend there to cube top − 0.2 cm,
then sweep toward the target along that same line at this fixed height.

## PushT-v1

Use the gray and red stem directions from the first overhead frame. The red crossbar end nearest the
pedestal is hard to reach at block height. Choose the contact on the other half of the T, at larger
robot-frame X, and sweep that feature so the stem rotates toward gray. From the reset pose approach bare
table about 3 cm outside the actual red boundary at half the T's thickness. Batch this first approach with
the tangential 10 cm push. If the stick stops on the red top, shift outward onto bare table, descend there
and immediately push in. Use the exposed stem from the new view to plan the next stroke toward the saved
gray target.

## PutCarrotOnPlateInScene-v1

For this fixed 512px Bridge camera use the visible [row,col]: r=(row-256)/100, c=(col-256)/100, d=1+.158*r.
Table-object XY in mount cm is about [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Segment “thick orange
carrot body” and “yellow plate” once. Apply the formula to both box centers and save XY to 0.1 cm. Move
above the carrot at the starting height with the jaws across its width. Descend to z≈2 cm and close.
On contact carry straight to plate center at z≈4.5 cm and open.

## PutEggplantInBasketScene-v1

In the fixed 512px sink view take the eggplant body's [row,col]. With r=(row-256)/100, c=(col-256)/100,
d=1+.157*r, approximate its XY at z=8 cm by [(29.2+3.68*c-4.25*r)/d,(-7.75-8.2*c-5.22*r)/d]. Segment
“eggplant” and “yellow basket” once. Save the basket floor center from this view: apply the same formula,
then convert (x,y) to (-.3+.774*(x+.3), -21+.774*(y+21)). Move above the eggplant at the starting height.
Descend to z≈8 cm and close even if contact stops the descent. Lift clear of the rim, move over the saved
basket XY, lower to z≈18 cm and open.

## PutSpoonOnTableClothInScene-v1

For this fixed 512px Bridge camera use the visible [row,col]: r=(row-256)/100, c=(col-256)/100, d=1+.158*r.
Table-object XY in mount cm is about [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Locate green handle midpoint
and towel center in the opening view. Batch a direct move to the handle at z≈2 cm and close immediately. If
needed for a crosswise grip, set the closest equivalent jaw yaw on that first move. At handle height avoid a
second downward probe. On contact, carry in one diagonal to towel center at z≈3–4 cm and open there.

## RollBall-v1

Strike with the forward-moving OUTSIDE finger, not the open gap. Measure the ball-to-goal line. From the
starting view, identify the leading fingertip and offset the TCP laterally by its measured displacement, so
that finger is centered on the back of the ball. Go directly to a clear point 6 cm behind ball center at
table+3.5 cm. Keep the leading edge sweeping in the goal direction: translate in a single follow-through and
turn the wrist about 0.8 rad during that same move in the sense that ADDS fingertip velocity toward the
target. Stay in contact through the equator and let the ball roll freely.

## SO100GraspCube-v1

Save the start pose. Segment the red cube in the 512px base view; its box is [r0,c0,r1,c1]. Let r=(r0+r1)/2-256,
c=(c0+c1)/2-256, d=1+0.00275*r. Table XY in cm is F=[(-10.44+0.0626*c+0.1358*r)/d,(-23.04+0.0938*c-0.173*r)/d].
Batch opening 0.25 and move_to([F.x,F.y,9.5]). Descend toward the cube at that XY; if the visible finger gap is
off center, add F(cube)-F(gap) to the measured XY. At 1.4–1.6 cm close to 0.12. Lift back to the saved start
with this grip; check whether the cube rises before considering reopening.

## StackCube-v1

Segment “red cube” and “green cube” once; save XY to 0.1 cm and green's top height. Use two batches with
the starting wrist. First move directly to red XY at table height + 2.1 cm, then close. On contact move
diagonally to green top + 3.5 cm, aiming inside green's top toward red by up to 1 cm in x and 1.5 cm
in y. Include full opening in this second batch to release within {max_episode_steps} steps.

## StackGreenCubeOnYellowCubeBakedTexInScene-v1

For the fixed 512px camera use cube centers: r=(row-256)/100, c=(col-256)/100, d=1+.158*r. Table XY in mount cm
is [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Segment “green block” and “yellow block” once. Use the two
box centers to save XY to 0.1 cm, then move above green with the starting wrist. Lower to its mid-height
and close. Both cubes are 3 cm tall. Lift enough to clear yellow, transfer to its saved XY, then lower
until green rests on yellow and open in the same batch before step 60.

## StackPyramid-v1

Segment “green cube”, “red cube” and “blue cube” separately in the first view; save their XY. Keep green
fixed. Place red one cube width beside green with the other axis aligned, open fully, then lift. Verify the
two base masks once by color. As soon as the cubes are adjacent, grasp blue at its saved XY. Place it over
the midpoint of the two measured base centers at base top + half a cube edge, then open.

## TwoRobotPickCube-v1

Pass red to the right by sliding it along the table with one left finger. From the clear initial view choose
the finger on the side of the jaw opening with the clearest path. Place that finger's inner edge just behind
red on its left-arm side at mid-height, compensating for the visible finger-to-TCP lateral offset so the
finger itself overlaps the block. Sweep through red toward the right arm by 20–24 cm and clear left
sideways. Remeasure red near the receiver, then RIGHT in one batch: descend directly to mid-height, close,
carry diagonally to the measured green goal with the same grasp. That carry doubles as the lift.

## TwoRobotStackCube-v1

The ring has radius 6 cm and cubes are 4 cm wide. Pick a green landing point inside the edge nearest its
start with enough clearance from blue. Right acts first: get the green center from the center of its TOP
FACE in the clearest view, not visible sidewall median; directly grasp at table+2 cm, close and in one
diagonal carry set green onto the chosen spot. Open at table+2.2 cm and clear right. Left directly grasps
blue at mid-height, closes and sets it on the new green top, then opens. Leave the inactive arm at reset
until its turn. Use only XYZ on both wrists.

## UnitreeG1PlaceAppleInBowl-v1

Measure the apple and bowl in the head view. The grasp that starts transport is a direct right reach to
equator +3 cm toward robot left at RPY [1.9,-0.15,2.02], then full closure. Maintain that grip throughout
transfer. Roll the palm under the fruit only slightly, toward 2.2 rad while translating; a large roll on the
first carry can dislodge the apple. Carry low in one diagonal toward the near rim, keeping the apple
cradled, then go just above bowl center. Open once over the bowl, withdraw the hand vertically at least 13
cm above its bottom. If the apple slips, approach its new position from below the equator before closing
again.

## UnitreeG1TransportBox-v1

Bring the box within reach of both hands before a two-handed lift. From the head view locate the box top and
the edge of its CURRENT table nearest the torso. First use the right palm against the outboard side to slide
the carton a short distance inward WHILE its bottom stays supported. Keep the measured wrist on the
approach. Remeasure the two end faces after this slide. Send left to the now-nearer end, right to the
opposite end below the top, and squeeze them with open palms. Lift the ends a little at a time while
maintaining contact, carry across to the other table and release. Avoid reaching the left palm all the way
across the body to the far outside corner.
