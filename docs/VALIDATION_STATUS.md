# Validation Status

This document reconciles the current validation evidence without changing algorithms, thresholds, feature tables, masks, or outputs.

## Classification

- Synthetic OOP validation: `controlled_implementation_validated`
- Manual crop Z-disc mask validation: `pilot_only_not_confirmatory`
- Manual full-image Z-disc mask validation: `pilot_only_not_confirmatory`
- Full-image patch mask validation: `pilot_only_not_confirmatory`
- Spacing: `exploratory_low_yield`

## Synthetic OOP Validation

- Synthetic examples: 72
- Clean angular error median: 0.31775328371710465 deg
- Clean angular error max: 0.317754176975825 deg
- OOP monotonicity low > medium > high: True
- Recovered OOP by disorder: {'high': 0.844289756635255, 'low': 0.998193159593016, 'medium': 0.9086115051157587}
- Caveat: Controlled synthetic recovery validates implementation behavior only; it does not prove real-tissue biological validity.

## Manual Crop Z-Disc Masks

- Masks: 40
- Z-disc-labeled masks: 12
- Orientation pairs: 9
- Median angular error: 20.71316363698199 deg
- OOP medians: {'zdisc_labeled': 0.5758803184165707, 'empty': 0.4046618750655299, 'ignore_only': 0.6366503686554358, 'mixed': None}
- Caveat: User-drawn sparse crop masks are pilot annotations, not blinded expert validation.

## Manual Full-Image Z-Disc Masks

- Full images: 12
- Labeled images: 7
- Orientation pairs: 7
- Median image-level angular error: 55.84144487595246 deg
- OOP medians: {'zdisc_labeled': 0.40462735723652, 'empty': 0.2396184453964924, 'ignore_only': None, 'mixed': None}
- Caveat: Sparse local manual masks are mismatched to global image-level orientation/OOP metrics.

## Full-Image Patch Mask Validation

- Patch rows: 2700
- Z-disc-labeled patches: 261
- Orientation pairs: 183
- Median patch-level angular error: 47.712762943687494 deg
- OOP medians: {'zdisc_labeled': 0.0720067156585264, 'empty': 0.0743168377073267, 'ignore_only': None, 'mixed': None}
- Spearman rho: -0.027195857318028028
- Caveat: Pilot patch-level comparison did not support OOP validation against the current sparse Z-disc masks.

## Spacing

- Status: `exploratory_low_yield`
- Valid spacing patches: 14
- Statement: Spacing is preserved as an exploratory low-yield descriptor and is not a primary endpoint.

## Overall Decision

- OOP implementation: `validated_on_controlled_synthetic_data`
- Real-tissue OOP: `not_expert_validated_unresolved`
- Manual Z-disc masks: `useful_pilot_annotations_not_sufficient_validation`
- Next required evidence: `expert_or_user_manual_organisation_orientation_annotation_not_more_zdisc_masks`
- Clinical/statistical analysis: `downstream_and_caveated_until_validation_route_is_clarified`

The OOP/orientation code behaves correctly on controlled synthetic striations, but current manual Z-disc mask pilots do not validate OOP on real archival tissue. Real-tissue OOP therefore remains unresolved and needs a better expert/manual organisation-orientation validation route before strong claims.
