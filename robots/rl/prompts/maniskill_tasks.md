# ManiSkill task instructions

## DrawSVG-v1

Clear the pen from the outline and keep that base-camera frame as a map. Segment the black path once.
Back-project the centerline pixels at both ends of the first visible edge in that frame, then batch hover, lower
to canvas height and trace. For the next edge reuse the last endpoint and back-project the next corner. Join
consecutive edges at drawing height, lift across gaps, and cover interior strokes. Compare the ink with the map
after each group of strokes.

## DrawTriangle-v1

Move the pen up and clear of the outline. Use that unobstructed base view for all three corners. This triangle
is equilateral. Back-project the two clearest black centerline junctions. Check the third against the
equilateral apex on the visible side and verify all three side lengths agree within about 1 cm before touching.
Hover over the verified corner, lower vertically, and follow each edge through its midpoint to the next corner
at canvas height.

## FMBAssembly1Easy-v1

Open the base view by shifting the raised hand aside once. Identify the green bridge in the clear frame. Use
the narrow rail's two ends and its midpoint, not the median of feet and rail together. Set the jaw closing
line across that rail during the diagonal approach. Descend between the feet; at a stalled descent with the
pads straddling green, close there and lift. If it stays on the table, use the clear frame to move one pad
width along the rail and turn crosswise before retrying. Stand the carried bridge on its feet, regrasp from
above, fit the two feet into the slots around the yellow peg and open.

## InsertFlower-v1

Locate the white bloom, adjacent narrow stem and vase in a clear frame. Use a thumb–index side pinch just
below the bloom. Orient the open hand at [-1.67,-0.05,-1.57] over bare table. Estimate palm XYZ as neck
minus [10.3,2.8,-6.2] cm, then use the visible midpoint between thumb and index as the actual contact point.
Correct the palm by neck minus that observed midpoint before closure. Pinch and lift only 2 cm first; if the
flower stays down, correct the finger-gap offset from the new view. Once it rises, carry above the vase
mouth, turn stem downward, lower and open.

## LiftPegUpright-v1

Segment each colored half. Use the midpoint of their two centers for the grasp XY. Batch move directly to
the seam at table+2.5 cm, close, then lift and tilt together. For a shaft running along the robot X axis and
a top-down wrist near yaw -pi/2, target absolute roll 0.9 rad, pitch 0 and measured yaw. Aim the grasp
center at table+12 cm and only 5 cm toward the base. The shorter horizontal transport and larger commanded
tilt let the shaft reach vertical within 50 steps. Keep the grip closed.

## PegInsertionSide-v1

From the first unobstructed view save the orange tip, white shaft, hole center and insertion direction.
Obtain hole position by back-projecting the rim and using its center on the front plane. Grasp the white
side of the seam crosswise, then carry toward the hole immediately. Keep a signed vector from TCP to orange
tip. Apply the wrist yaw change to that vector; TCP target = hole center minus rotated vector minus 2 cm
along the inward normal. Set yaw and translation together in this move. Then advance along that normal. Use
the stored aperture while the arm occludes it; consult a second camera once only if contact is off-center.

## PickCube-v1

The wrist starts pointing down. Omit orientation_rpy from move_to to keep that orientation. Use the segmented
cube's x and y. For the grasp height, add half of cube_size_cm to table_height_above_base_cm. The initial wrist
pose is already a clear hover. Move directly to the grasp position, then close. Once is_grasped is true, carry
the cube centre directly to goal.xyz and hold it still with the gripper closed.

## PickCubeSO100-v1

Measure the red cube and green sphere at reset; save the sphere center from its contour and top minus
radius. Batch the preshape to 0.25, direct descent to cube center at table + half its height, and pinch to
0.12. Carry immediately. The SO100's nearly horizontal link can bind on a vertical lift with fixed wrist
pitch. For the carry only, let the wrist pitch open toward the rising arm: target grasp pitch minus 0.25
rad, retain measured roll/yaw. Aim at the sphere center plus measured closed TCP-minus-cube offset. Keep the
grip and let the arm settle.

## PickCubeWidowXAI-v1

Save green sphere center and red cube footprint from the initial view. Use only three commands for pickup
and delivery: one straight move from reset to cube mid-height, full closure, then a diagonal carry. The
reset pose is the hover; an added waypoint puts closure too late for the arm to settle. Keep measured wrist
on descent. Aim near the sphere center with a horizontal bias of only 0.5 cm toward the base, plus the
closed TCP-minus-cube offset. On carry retain measured yaw and pitch and reduce roll by 0.2 rad. After
arrival, keep the grip still.

## PickSingleYCB-v1

Locate the object and green sphere in the first image; save the target before the hand covers it. For a
small rectangular object pinch the middle of its two long sides at upper sidewall height, jaw line across
the narrower dimension. Choose the nearer equivalent yaw. Put this yaw and the full descent in the first
move, then close immediately. The initial pose provides the hover. Carry diagonally directly into the
sphere, subtracting the measured object-center-minus-TCP offset, and hold still. A separate lift on the spot
can consume the remaining episode.

## PlaceSphere-v1

From the initial base view, segment sphere and bin. Back-project the sphere mask's top 1 cm (z_range_cm); its XY
is the grasp XY. Use exactly that XY. The equator is halfway between table height and sphere top. From the clear
start batch move_to(equator) and close fully. After contact, batch one diagonal carry to the bin mask's XY at
bin rim top + sphere radius + 0.3 cm and open fully. If the ball is in the bin and the episode continues, hold
the wrist still with the open-gripper target.

## PlugCharger-v1

In the first view identify two distinct white parts: the small loose body with two pins and the larger fixed
outlet. Segment the small plug first and grasp its body crosswise with a top-down wrist. Batch approach,
descend and close; preserve measured pitch and roll for this pickup. Use the held plug to find the forward
pin direction. For the outlet use the center of the wall facing the pins, about 5 cm below its top. Carry
with the measured pin-to-TCP offset to stage 2 cm in front, turn the pins along the socket normal during
that carry, then insert straight. Inspect the face from a second camera only if alignment stalls.

## PokeCube-v1

From the base view measure the shaft midline, working tip, cube and target. Grasp about 1 cm from the shaft
midpoint toward its blunt end. First batch: hover 4 cm above the shaft top; descend at the same XY straight to
1.8 cm BELOW the top; close fully. These are the only approach waypoints. At finger contact use the
just-measured TCP z for both remaining moves. Batch a lateral alignment that brings the tip to the cube's near
face on the push line, then a push along that line into the circle. This alignment also tests the grasp. Keep
the grip and wrist still to settle.

## PullCube-v1

Keep the measured downward wrist orientation throughout the pickup and pull. Descend to the visible cube, close
and lift slightly. Pull toward the goal along the cube-to-goal line without a separate wrist turn.

## PullCubeTool-v1

Measure blue cube and both ends of the red tool. Grip the handle, orienting the jaw across it during
approach. Use the blade's INNER catching face and its farthest corner as offsets from the grasp TCP. Clear
the cube sideways first by half cube width plus the blade thickness. At that side clearance slide the blade
beyond the cube's far face, then bring it sideways behind the cube. These legs must not cross the cube. Pull
straight toward the base far enough for cube-to-base XY distance below 60 cm. Keep the blade face at cube
mid-height and broadside to the pull.

## PushCube-v1

Close the jaws and omit orientation_rpy for approach and push. Reach contact height just behind the cube on the
cube-to-goal line, then push along that line at the same height.

## PushT-v1

Use the initial base image as a fixed goal map. The gray printed target sits on the table: obtain its
corners with single-pixel back_project on the visible gray edges. Red mask is optional; locate its stem and
crossbar directly in that same image. Compute rotation from stem direction, including its sign. Approach a
red bar end on bare table at half-height, then push tangentially through it. If the hand covers red after
contact, shift the pusher laterally by 4 cm in the same low plane and use the newly exposed edge. Execute
the next push using the saved goal instead of repeatedly segmenting the shaded/covered target. Finish with a
centered translation if angles match.

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

Measure blue center B and red goal from the wide view. The ball radius is 3.5 cm. Draw the table-plane unit
vector u from B to the goal. Use the open or partly open hand at reset as a paddle. First action: diagonal
XYZ-only move straight to bare table behind B at B-7 cm*u, z=table+3.5 cm. No high staging pose or separate
jaw adjustment. Second action: one uninterrupted stroke toward B+28 cm*u, z=table+3.5 cm. Use the remaining
steps to let the ball roll; correct only after seeing its actual travel.

## SO100GraspCube-v1

Save the start pose. Segment the red cube in the 512px base view; its box is [r0,c0,r1,c1]. Let r=(r0+r1)/2-256,
c=(c0+c1)/2-256, d=1+0.00275*r. Table XY in cm is F=[(-10.44+0.0626*c+0.1358*r)/d,(-23.04+0.0938*c-0.173*r)/d].
Batch opening 0.25 and move_to([F.x,F.y,9.5]). Descend toward the cube at that XY; if the visible finger gap is
off center, add F(cube)-F(gap) to the measured XY. At 1.4–1.6 cm close to 0.12. Lift back to the saved start
with this grip; check whether the cube rises before considering reopening.

## StackCube-v1

The tabletop cubes have 4 cm edges. Use two batches. From the initial clear wrist, move to red's segmented XY
(round to 0.1 cm), z = table height + 2.1 cm, then close. On finger contact the lift and carry are one diagonal
move followed by full opening. Use green's measured top and the physical red half-edge of 2.0 cm: the release
TCP z is green top + 2.0 + 1.5 cm. For release XY aim inside green's top, toward red by up to 1 cm in x and 1.5
cm in y. Keep wrist orientation throughout. Open immediately on arrival to let red settle on green within
{max_episode_steps} steps.

## StackGreenCubeOnYellowCubeBakedTexInScene-v1

For the fixed 512px camera use cube centers: r=(row-256)/100, c=(col-256)/100, d=1+.158*r. Table XY in mount cm
is [(31.8+4.86*c-4.55*r)/d,(1.6-8.8*c-5.04*r)/d]. Start about 5 cm above green's top. Lower directly to its
mid-height and close. Both cubes are 3 cm tall. Lift just high enough to clear yellow and transfer at that
height. Center green over yellow and use the measured grasp offset to lower its underside onto yellow's top.
Batch that contact-height move with full opening before step 60.

## StackPyramid-v1

Choose the red and green base positions before moving. When one base cube is already stable, place the other
beside it and leave the pair in place. Pick up blue only after verifying the base, then release it over the
midpoint at the measured height of the two cube tops.

## TwoRobotPickCube-v1

Measure red and the green target in the two fixed views before moving. Left takes red with a direct descent
from reset and full closure, then carries diagonally to a handoff 1.5 cube widths into RIGHT territory at
one cube height. Open and clear left diagonally toward its base by three cube widths, not just one: leave
the right arm a free downward path. Right descends directly to the cube upper half and closes; skip the
extra hover. Carry to the saved target in one move. Keep each arm's reset wrist throughout.

## TwoRobotStackCube-v1

Measure both cube centers and ring first. If blue sits in the ring, left picks it directly at mid-height and
closes, then lifts only 5 cm to leave the ring clear. Right descends straight to green at mid-height and
closes. Place green on the near edge inside the ring in one diagonal carry and open. Left carries its held
blue onto the newly placed green and opens. Use each arm's reset wrist and use the initial height as its
approach hover. Spend the 100 steps on carrying blocks rather than raising an idle hand or repositioning the
wrist.

## UnitreeG1PlaceAppleInBowl-v1

Segment apple and bowl in the head view. Reach with right hand directly to apple equator with TCP 3 cm to
robot left of its center, RPY [2.5,-0.1,2.02]; this rotates the fingers under the apple as they close. Close
fully when the palm is beside the fruit. Lift one radius in the direction of the bowl, preserving that
wrist. Compare the apple to the table: if still down, open, place the finger gap around its visible sides
and reclose; if it rose, follow through to bowl center, release and clear the bowl vertically by 13 cm.

## UnitreeG1TransportBox-v1

Use the head view to measure the 36 x 24 cm top rectangle and table height. For each long-axis end, get the
grasp at its center biased 7 cm toward robot left, 7 cm below top. Assign the end nearer the right palm to
right, the other to left. Move each palm to its own end early: right contact then left contact, with no trip
to the empty table or backward excursion. Rotate each open palm inward to face its opposite palm during
approach. If one reach stalls, command the other arm before retrying. With both hands squeezing, raise each
end just 2 cm alternately, carry toward the other tabletop maintaining pressure, and release there.
