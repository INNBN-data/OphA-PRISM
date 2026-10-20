# Endpoint definitions

Class 1 denotes the adverse outcome or higher observed operative burden. The unit of analysis is the patient and index aneurysm.

| Target | Definition and implementation |
| --- | --- |
| VISION | Adjudicated persistent new visual morbidity assessed at 75–120 days and confirmed at six months; supplied as `vision_event`. The reported rule includes a decline of at least two Snellen lines or 0.20 logMAR, or qualifying field deterioration. Exact perimetry criteria must agree with the source adjudication protocol. |
| ACCESS | Supplied fixed exposure-burden label, `access_high_burden`. The secondary transparent definition is at least five of eight recorded maneuvers. |
| CONTROL | Supplied fixed vascular-control/decompression label, `control_high_burden`. Components include proximal cervical control, temporary clipping, suction decompression, and adjunctive control. |
| RECON | More than one clip or a fenestrated, tandem, or booster configuration; checked against `recon_label` when source fields are available. |
| CURE | Failure of favorable angiography, preserved OphA patency, or absence of VISION morbidity. Derived and checked against `cure_failure`. |

## Angiographic and composite definitions

Let $G$ be RROC grade, $A$ the accepted-remnant flag, $L$ OphA patency loss, and $V$ visual morbidity. For known components,

$$
F=I(G=1)\lor[I(G=2)\land I(A=1)],
\qquad
T=F\land I(L=0),
\qquad
C=\neg[T\land I(V=0)].
$$

Here $F$ denotes favorable angiography, $T$ technical success, and $C$ CURE failure. RROC I alone means complete occlusion. Accepted RROC II is a favorable remnant; RROC III is a residual sac. Missing components retain uncertainty unless an observed component determines the composite. Saved labels are checked against these rules.

The reported 221 favorable angiographic results comprise 210 RROC I and 11 accepted RROC II results. Reported technical success is 220; CURE success is 211. These counts require confirmation against the controlled source records.

## Operative targets

The eight ACCESS maneuvers are complete clinoidectomy, optic-canal unroofing, optic-strut drilling, distal dural-ring opening, falciform opening, optic-nerve mobilization, cervical ICA exposure, and suction decompression. The count-based rule requires known values for all eight components.

ACCESS and CONTROL consume fixed cohort-derived labels. The reported original construction assigned 18 indeterminate cases to low burden. Predictions therefore characterize recorded operative practice conditional on these definitions. Shared components and surgeon-dependent decisions preclude treating the five targets as independent anatomical dimensions.

The code validates endpoint consistency; it does not perform clinical visual adjudication or reconstruct the original latent clustering.
