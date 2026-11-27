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

After clearing the raised hand sideways, inspect the exposed green outline. Pick the outer free end of ONE
thick green foot, with open space on both sides; the internal gap beside the connecting rail can admit only
one finger. Point back-project that foot top in the clear frame. Keep jaws fully open until they straddle
the foot. Align across its short width in the same move as approach, descend only 1 cm below the green top,
then close even if the last descent is partial. Confirm green follows a short lift; transport to the
fixture, reorient and regrasp, then lower both green feet into the board slots flanking yellow.

## InsertFlower-v1

First move the wrist clear to see the bloom and vase together. Secure the thick white bloom at its base
using the open palm as a scoop: sweep in along the table with the thumb on one side and finger pads on the
other. Start with the palm about 10 cm behind the bloom and 6 cm above it, at RPY [-1.67,-0.05,-1.57]. Close
to 0.05, holding that closure. If the bloom rises with the hand, immediately translate the held bloom
diagonally toward the vase mouth (2–3 cm above the rim), then turn the stem down, lower and open. Use fresh
view_env after lift; avoid another squeeze or reopening before the transfer.

## LiftPegUpright-v1

Measure both colored halves and aim at the midpoint at table+2.5 cm. Begin the roll DURING the approach: for
a rod along robot X and wrist yaw near -pi/2, use absolute roll 2.60 rad, pitch 0, measured yaw, so the jaw
closing line remains crosswise. Close fully, then pivot toward absolute roll 1.25 rad, pitch 0, same yaw,
while lifting the midpoint to table+12 cm and translating about 10 cm toward the base. The peg remains
pinched throughout. Starting the pickup slightly tilted leaves less wrist rotation after grasp.

## PegInsertionSide-v1

From the clear wrist camera measure hole and two ends of the shaft. Contact on the socket's FRONT surface is
still outside the hole. Grasp white crosswise and batch the close with a diagonal carry toward the opening,
keeping the orange tip pointing in. Use a rim-based center in both lateral axes; back-project a rim point at
either side and use their midpoint, not a bright mark inside the recess. Stage the tip 2 cm outside. On
first blocked entry withdraw a finger width and adjust the LOW tip up about 0.8 cm, then advance along the
socket axis by the visible depth of the block. Do not repeat the blocked line.

## PickCube-v1

The wrist starts pointing down. Omit orientation_rpy from move_to to keep that orientation. Use the segmented
cube's x and y. For the grasp height, add half of cube_size_cm to table_height_above_base_cm. The initial wrist
pose is already a clear hover. Move directly to the grasp position, then close. Once is_grasped is true, carry
the cube centre directly to goal.xyz and hold it still with the gripper closed.

## PickCubeSO100-v1

Measure both objects. Preshape 0.25, descend directly to cube mid-height, close to 0.12. The closed pinch
can carry the cube. For the lift, keep roll and yaw and INCREASE pitch from the measured closing pitch by
about 0.2 rad, giving the arm room to raise its TCP without drifting sideways. Use one move to the goal's
lower interior (sphere center minus 0.5 cm height plus measured TCP-minus-cube offset). On arrival keep the
same opening to settle. If the servo stalls, use the actual cube-to-goal displacement from the new view.

## PickCubeWidowXAI-v1

Measure red and green before moving. Starting with open jaws, descend from the reset height directly onto
red at mid-height and close fully. Once the fingers touch, use one diagonal carry. Aim at the near/lower
interior of the green sphere: from its center move 0.7 cm toward the base and 1 cm down, then add the
measured TCP-minus-cube offset. Keep the measured closing wrist for the carry (XYZ only). The cube must stop
inside the sphere: on arrival keep opening 0 for a settling action. Do not spend the final steps chasing the
exact center.

## PickSingleYCB-v1

Descend directly from the high reset pose, pinching the small object crosswise on its upper sidewalls. Batch
this approach and full closure. The raised hand may hide the green goal at reset. Immediately after descent
use the CURRENT base-camera image, where the hand has uncovered the marker. Point-segment its green center
or back-project a small patch inside it; allow for the sphere radius when estimating its center. Carry
directly from grasp to that measured target, keeping the object's measured offset. An intermediate lift to
an arbitrary height uses up the 50-step episode.

## PlaceSphere-v1

From the initial base view, segment sphere and bin. Back-project the sphere mask's top 1 cm (z_range_cm); its XY
is the grasp XY. Use exactly that XY. The equator is halfway between table height and sphere top. From the clear
start batch move_to(equator) and close fully. After contact, batch one diagonal carry to the bin mask's XY at
bin rim top + sphere radius + 0.3 cm and open fully. If the ball is in the bin and the episode continues, hold
the wrist still with the open-gripper target.

## PlugCharger-v1

Take the small body crosswise just behind the pins with a top-down pinch. Record the rim of the fixed
receptacle and both pin tips in the clearest view. With the held plug almost level, stage in front of the
front wall and advance. If travel stalls at the face, keep wrist fixed: back off 0.8 cm, adjust height so
BOTH metal pin centers align with BOTH hole centers, then push straight 2–3 cm past the face. Follow the
paired-hole midpoint, not the plug body's center or the top edge. If the first correction stalls, change the
lateral offset 0.2 cm and repeat before taking more pictures; pin alignment needs millimeters.

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

Grasp the thin straight handle near its FREE end, about 2 cm in toward the elbow from that end; gripping the
elbow leaves too little reach. Keep wrist close to its reset yaw and close fully around mid-height. Plan a
low hook using the inner blade tip. At cube mid-height take a dogleg: first slide the TCP sideways away from
the cube by about 8 cm, then forward until the blade is a little beyond the far cube face, then sideways
until it overlaps the cube. Pull straight back toward the base by 18–22 cm. The side slide is the grasp
check. Keep all four strokes close to the table.

## PushCube-v1

Close the jaws and omit orientation_rpy for approach and push. Reach contact height just behind the cube on the
cube-to-goal line, then push along that line at the same height.

## PushT-v1

Match the RED and gray stems in the initial overhead image before touching. Use the two endpoints of each
long crossbar and a single pixel on each stem to get signed rotation. Use the edge that is both reachable
and farthest from the T's center; prefer the side toward the initial TCP when the opposite end is near the
pedestal. Approach through bare table 3 cm outside the contact face at block mid-height; follow with a 10 cm
tangential sweep. A stall during approach already places the pusher at contact: start the sweep immediately.
Re-measure the new angle and use one more low push toward the fixed gray outline.

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

Use the ball-to-goal line on the table and the regulation ball radius 3.5 cm. Strike with the OUTSIDE of a
finger pair, rather than trapping the ball between open jaws. Close fully while the wrist is clear; approach
a launch point only 4.5 cm behind ball center, at table+3.5 cm, wrist as measured. For a long goal, keep
advancing in that same line through the ball as far as the arm can comfortably reach (about 40 cm forward
travel), raising the TCP by one radius at the end to release the rolling ball. A stalled approach already
within one radius of launch should immediately start the sweep.

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

Use a low table handoff at the shared midline. Left goes directly from reset to red mid-height, closes, then
transports it just one half-width into the right workspace at table+2.5 cm. This transport also tests the
grasp; skip separate height adjustments. Open only to 0.65 and clear left diagonally toward its own base by
10 cm to free the right path. Right directly closes on the exposed red upper half. Locate the green sphere
with the unobstructed camera while the arms are low, then carry the cube there. Use XYZ only throughout.

## TwoRobotStackCube-v1

Execute serially, one arm per batch. The blue top block may occupy the ring. LEFT first: descend directly at
blue center height, close, carry blue 6 cm toward left's own base while raising it just one cube width. This
clears the ring for green. RIGHT next: descend directly from reset to green center, close, carry diagonally
to the edge inside the ring nearest green at one cube height. Open and clear right. LEFT finishes by moving
blue to rest on measured green and opening. Start both descents from their initial high wrists. Avoid
consuming a forced move of either waiting arm.

## UnitreeG1PlaceAppleInBowl-v1

From the head view locate apple and bowl first. Right hand approaches at apple EQUATOR, not a separate high
hover: target apple center plus 3 cm in +Y, RPY [1.9,-0.15,2.02]. The fingers must curl; with the apple
beside the palm issue set_gripper right opening 0 even if the approach reported stalled. From that grasp go
straight a short distance toward the bowl while lifting 4 cm and rolling the wrist toward 2.6 rad. If fruit
moved with it, carry over bowl, release and clear; if fruit stayed, reopen and put the palm on the side of
the apple toward the bowl for the next closure.

## UnitreeG1TransportBox-v1

Avoid a half-turn of either wrist while reaching the carton. Use each hand's measured initial RPY on its
first translation. Locate the near and far ends of the box along its long axis. Send RIGHT directly to its
near end at 4 cm below top, then LEFT directly to the opposite end, alternating arms even if the right reach
stalls. When moving into contact choose the smallest inward wrist turn from the actual palm pose. Keep both
open hands squeezing; alternate small lifts and short lateral carries toward the other table, then release
and let the box settle.
