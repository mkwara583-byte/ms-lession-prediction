Predicting MS Lesion Segmentation and Tissue Vulnerability from Brain MRI

A two-stage machine learning pipeline on longitudinal multiple sclerosis brain MRI:

Stage A — a 2D U-Net that outlines lesions on a FLAIR scan.
Stage B — voxel-level models that estimate which healthy-looking tissue became lesion by the patient's next scan.

This is an independent methods demonstration on a small public dataset. It is not a clinical tool and has not been validated for any medical use.

Data

MSLesSeg (Guarnera et al., Scientific Data 12:920, 2025), DOI 10.6084/m9.figshare.27919209, released under CC-BY-4.0.

Only the training portion was available for download: 53 patients, 93 FLAIR series (28/15/5/5 patients with 1/2/3/4 timepoints). The 22-patient test set was not included. The count of 28 single-timepoint patients initially looked wrong against the paper's stated 50 — it resolves as 50 minus the 22 test patients, confirming nothing was missing from the download.

Per the dataset paper, all scans are skull-stripped (BET) and affine-registered to the MNI152 template at 1 mm, shape 182×218×182, LAS orientation, with one expert-validated lesion mask per timepoint. Every one of those claims was checked against the files rather than assumed.

Raw scans are not included in this repository. Download them from the link above.

Data audit and cleaning
Check	Result
Shape / voxel size / orientation	Identical across all files: (182, 218, 182), 1 mm, LAS
Every FLAIR has a mask; masks binary	Passed (values 0/1 only)
Skull-stripped	Passed — minimum zero-voxel fraction 0.516, background exactly 0
Lesion voxels outside the brain	0
Raw intensity range	Scan medians ~170–1100, maxima ~326–2417 → per-scan normalization required
The three outlier scans, and how they were resolved

Three scans (P44 T1, P45 T1, P53 T1) had nonzero extents of ~178×213×176 voxels (~3.3M voxels) against ~146×181×153 (~1.9M) typical — nearly double the brain volume of a normal scan.

The first hypothesis was that brain extraction had failed on these scans, leaving skull, scalp, or neck tissue attached. Visual inspection rejected it. My observation on looking at the scans: only the brain was showing, the background was black, and no extra tissue surrounded the brain. Whatever was inflating the volume, it was not leftover anatomy.

A second hypothesis came from what the images actually looked like: P45 and P53 both had enlarged dark areas resembling a curved X shape on a mid-brain slice, and P44's brain tissue was noticeably lighter than the others. The curved X is the lateral ventricles, which appear dark on FLAIR because the sequence suppresses fluid signal — so enlarged ventricles, consistent with brain atrophy, seemed a plausible explanation.

Quantification rejected that too. The dark-voxel fraction in those scans (0.009–0.021) was lower than typical (0.03), and larger ventricles cannot add nonzero voxels in the first place. The remaining explanation consistent with the measurements is over-scaling during affine registration: the extents are ~15–22% larger per axis, and 1.22 × 1.18 × 1.15 ≈ 1.66, which matches the observed ~1.7× volume ratio. (Hypothesis consistent with the measurements; not independently confirmed.)

P53 settled it. Its two timepoints overlap at Dice 0.708 with a brain-volume ratio of 0.55 — one brain sitting almost entirely inside a scan 1.8× larger. That pair cannot be compared voxel-by-voxel, so P53 T1 was excluded. P44 T1 and P45 T1 have no follow-up scan and never enter Stage B; they were kept for Stage A and flagged.

Across the remaining 39 follow-up pairs, brain overlap had median 0.968 and minimum 0.889. Three borderline pairs (P28 0.889, P20 0.902, P52 0.926) were kept and flagged for a sensitivity check.

Final cohort: 92 FLAIR scans, 53 patients, 24 with follow-up, 39 consecutive scan pairs.

Preprocessing
Brain mask = voxels > 0 (valid because scans are skull-stripped), hole-filled.
Per-scan normalization: clip the top 0.5% of brain voxels, then z-score within the brain mask. Required because raw intensities vary roughly 7× across scans — the dataset was collected on mixed 1.5T and 3T scanners with no unified protocol.
Padded 182×218 → 192×224 so each side divides by 32 for the U-Net's downsampling path.
A patient-wise split table was written before any modeling, with a fixed seed.
Stage A — lesion segmentation

Architecture. 2D U-Net with an ImageNet-pretrained ResNet34 encoder. Input is three adjacent axial slices stacked as channels ("2.5D"). Loss is Dice + focal. AdamW, lr 3e-4, random left-right flip and intensity jitter, ~30% lesion-free slices per epoch, early stopping on validation Dice.

Leak-free design. Patients were split into four roles: train_always (23), val (6), long_A (12), long_B (12). Two models were trained:

Model	Trains on	Predicts
Model 1	train_always + long_B	long_A (+ val)
Model 2	train_always + long_A	long_B

Every follow-up patient is therefore predicted by a model that never saw them — required for Stage B's predicted-mask experiment to be honest.

Results (69 held-out scans)
Method	Dice	Lesions found	Small lesions found (<30 vox)	False blobs/scan
Threshold baseline (z > 1.75, min blob 40)	0.331	28.6%	13.7%	14.0
Threshold baseline (z > 1.75, min blob 5)	0.327	45.0%	23.9%	80.2
U-Net	0.686	78.0%	47.3%	21.1

By group: long_A 0.667, long_B 0.711, val 0.659.

The U-Net beat the Dice baseline in 65 of 69 scans; mean per-patient Dice gain +0.356 (95% patient-bootstrap CI 0.284–0.430).

Two baselines are reported because the minimum blob size barely moves Dice (0.331 → 0.327) while changing detection substantially (28.6% → 45.0%). Small lesions contain too few voxels to shift an overlap metric, which is why lesion-wise detection is reported alongside Dice.

On the baseline's own ceiling. A grid over six intensity thresholds and seven minimum blob sizes found the best combination at z > 1.75 with minimum blob 40, at Dice 0.196 on the tuning patients. Dice collapses to 0.080 at a threshold of 2.0 and 0.011 at 3.0, and falls to 0.116 at 1.25 where false positives take over. Lesion and healthy-tissue brightness overlap enough that no single threshold separates them — which is the empirical argument for a model that uses shape, size, and context rather than intensity alone.

What the segmentations look like

Reviewing held-out predictions against expert annotations, my observation was that the images showed no defects in the brain imaging, and the blue and red outlines aligned well but not perfectly. Near-agreement with visible edge disagreement is the expected result, for reasons that are structural rather than fixable: lesion boundaries are gradual rather than sharp (partial volume effects, compounded by ~2.6 mm raw slices resampled to 1 mm), the expert outlines with judgment while a model outputs a threshold on a probability, and different expert raters do not draw identical boundaries in the first place. No method scores 1.0 against a single rater.

A second observation turned out to be more diagnostic: the middle image showed red-outlined lesions in the centre of the brain with no cyan perimeter around them, and each of the four worst-case scans had one untraced red lesion. That suggested the model was systematically missing centrally located lesions.

Testing it across all 1,932 held-out lesions did not support the hypothesis. Found rates were 77.3% (central third), 82.9% (middle), 73.8% (outer), and missed and found lesions had identical median distance from the brain centroid (47.4 vs 47.3). The four scans that prompted the idea were the worst-performing scans, selected by Dice, each shown on a single slice — too small and too biased a sample to generalize from. What the full analysis found instead was that size and contrast drive detection.

Error analysis (1,932 held-out lesions)
Size (voxels)	1–5	6–10	11–30	31–100	>100
Found	21.1%	39.4%	57.6%	81.8%	93.2%
Missed lesions are smaller (median 28 vs 82 voxels) and fainter (median z 1.0 vs 1.4) than found ones.
84% of misses have a maximum predicted probability ≤ 0.05 — the model does not see them at all, so lowering the 0.5 threshold is a weak lever (at most ~16% of misses are near-misses).
Only ~54% of misses are ≤30 voxels; ~11% are >100 voxels, so size does not explain all of them.
~82% of the U-Net's false blobs are ≤30 voxels; blobs >100 voxels are only ~10% false (~1 per scan).
Stage B — tissue vulnerability

Setup. For each of the 39 consecutive pairs, candidate voxels are brain voxels at time 1 that are not expert-labeled lesion. The label is 1 if that voxel is lesion at time 2. All positives were kept plus a random 1% of negatives, with weights restoring the true base rate.

Features (8). Distance to nearest lesion, distance to ventricle, distance to brain surface, lesion density in ~5 mm and ~10 mm windows, FLAIR intensity (raw and smoothed), and the scan's total lesion load.

Two feature versions: computed from the expert time-1 mask, and from the U-Net's predicted time-1 mask (from the model that never trained on that patient).

Validation: grouped 6-fold cross-validation by patient. Metric: weighted AUPRC, judged against the base rate.

What the labels actually look like

Before modeling, follow-up pairs were reviewed visually. My observation on the first pair: the green area (voxels that became lesion) was around the red area (existing lesions), thin in some spots but thick in others.

That observation shaped the whole of Stage B. Quantified across all 39 pairs, 88.9% of new-lesion voxels are non-isolated — within 2 mm of an existing lesion, or in clusters smaller than 3 voxels. Only 11.1% belong to isolated new lesions (241 clusters, present in 36 of 39 pairs and in all 24 patients). The median lost/new ratio is 0.87, meaning nearly as many voxels disappear between scans as appear.

A thin rim that both gains and loses voxels is more consistent with annotation and registration noise than with biological growth; thicker patches are more plausibly real expansion. The consequence for evaluation is direct: a model scored only on "all new voxels" would look good largely by predicting proximity to existing lesions. Three evaluation views were therefore used — all new voxels, isolated lesions only, and far-field (candidates >2 mm from any lesion, where proximity cannot do the work).

Ventricle masks

Distance-to-ventricle requires a ventricle mask, built per scan: voxels dark relative to the scan's own median, restricted to a population-derived core (voxels dark in ≥50% of scans) dilated by 8 voxels.

The first version failed on one scan — my observation: the cyan tracing of the butterfly-shaped ventricle is missing on the first panel, which was the scan with the smallest computed ventricle volume, despite the ventricles being clearly visible in the image. A fallback threshold adapted to that scan's own CSF intensity resolved it, after which all panels showed cyan tracing. Masks were not used until every one had been visually confirmed.

Results

Base rate: 0.00271 (about 1 in 370 candidate voxels converts).

Features	Model	All new voxels	Isolated lesions only	Far-field (>2 mm)
Expert mask	distance only	0.1328	0.0003	0.0116
Expert mask	logistic regression	0.1685	0.0006	0.0187
Expert mask	LightGBM	0.1565	0.0008	0.0125
U-Net mask	distance only	0.1611	0.0036	0.0297
U-Net mask	logistic regression	0.1930	0.0028	0.0339
U-Net mask	LightGBM	0.1601	0.0023	0.0335
—	base rate (random)	0.00271	0.0003	0.00112

Removing the three borderline pairs changed scores by a few percent with the same ordering.

Per patient (expert-mask features), LightGBM beat distance-only in 22/24 patients (all new voxels), 24/24 (isolated), and 20/24 (far-field), with bootstrap intervals above zero in every view.

What drives the predictions

Ablation (single run, faster LightGBM setting, no confidence intervals):

Config	All new	Isolated	Far-field
Full model (8 features)	0.1596	0.0008	0.0124
Drop FLAIR brightness	0.1168	0.0004	0.0057
Drop ventricle distance	0.1383	0.0007	0.0105
Drop brain-surface distance	0.1613	0.0007	0.0132
Only lesion proximity + density	0.1189	0.0003	0.0054
Distance to existing lesions explains most of the signal — consistent with the 88.9% non-isolated share seen in the label review.
FLAIR brightness is the largest contributor beyond proximity: removing it cuts far-field AUPRC by 54%, down to roughly the proximity-only score. Mechanism unknown — could be genuine pre-lesional change, or boundary ambiguity in the annotations.
Ventricle distance helps modestly (~13–15%).
Brain-surface distance adds nothing in the ablation, despite ranking equal to ventricle distance in SHAP. SHAP importance and predictive contribution are not the same thing.

Calibration. Average predicted risk 0.00182 vs observed 0.00271. The top decile is reasonably calibrated (0.0167 predicted vs 0.0209 observed), but deciles 3–9 are overconfident about low risk. Ranking is unaffected; raw probabilities should not be quoted as absolute risk without recalibration.

Example risk map (P13 T1→T2, slice 99, model trained without P13): the top 1% of risk voxels were 34.8× enriched in voxels that actually converted, and the top 5% contained 60.5% of that slice's new-lesion voxels. One slice chosen for high lesion activity — an illustration, not a performance estimate.

An unexpected finding

Features computed from the U-Net's time-1 mask scored as high or higher than those from the expert mask, in 19–21 of 24 patients across all three views. This was the opposite of what was expected: the predicted mask was supposed to be the degraded version, and the gap between them was supposed to measure the cost of segmentation error.

The diagnostic explains why: 19.9% of voxels that later became lesion were already inside the U-Net's time-1 mask, versus 0.11% of ordinary candidate voxels — roughly 180× enrichment. Among candidates inside the U-Net mask, 32.7% converted (base rate 0.27%). Since candidates exclude expert-labeled lesion by construction, these are voxels the U-Net called lesion and the expert did not.

Possible explanations, not yet separated:

Boundary placement differs between timepoints (the time-2 outline is more generous than time-1), consistent with the 88.9% non-isolated share and the 0.87 median lost/new ratio.
The U-Net detects faint early abnormality the expert labeled only at the later scan.
The U-Net over-segments lesion edges in a way that resembles the time-2 outline.

Important caveat: because candidates are defined as "not in the expert mask," the U-Net's own detections count as healthy-looking candidates. A deployment-faithful evaluation would require candidates outside both masks, which would remove ~20% of all positives. That test has not been run, so the U-Net-feature advantage should not be claimed as a real improvement.

Limitations
Cohort size. 53 patients for Stage A, 24 for Stage B. Confidence intervals are wide.
Training set only. The dataset's 22-patient test set was unavailable, so there is no untouched final test set; Stage A uses patient-wise held-out groups instead.
Affine-only registration, applied per scan. Small misalignments can masquerade as lesion change; one patient (P53) was clearly affected and excluded.
Annotation noise. 88.9% of "new lesion" voxels are non-isolated, and the median lost/new ratio of 0.87 indicates substantial boundary churn in both directions. Part of the Stage B positive class is annotation or registration noise, not biology.
Candidate definition uses expert information, which is unavailable at deployment time (see above).
2D model. No full 3D context; the dataset paper's models were 3D.
Raw slice thickness was ~2.6 mm before resampling to 1 mm, so images are smoother than the voxel grid implies.
Single rater per mask, and scans came from mixed 1.5T and 3T scanners with no unified protocol.
LightGBM was not tuned. Pooled AUPRC slightly favored logistic regression; per-patient AUPRC favored LightGBM. Both are reported.
Comparison to the dataset paper's models is not like-for-like. Their 3D models averaged ~0.68 Dice on their own test set, but with different patients, splits, and evaluation. Similar numbers on different setups are not comparable.
Results are a methods demonstration, not clinical evidence.
Open items
Deployment-faithful Stage B evaluation (candidates outside both expert and predicted masks).
Ablation with load separated from the proximity group and patient-level confidence intervals — load is a per-scan constant and may act as a patient fingerprint with only 24 patients.
Test whether FLAIR brightness still contributes among voxels >5 mm from any lesion, which would distinguish pre-lesional change from boundary ambiguity.
How to run

All work is in a single notebook, run top to bottom in Google Colab:

00_explore.ipynb

Its sections, in order:

Data inventory and integrity checks (shape, orientation, masks, skull-stripping, timepoint alignment)
Preprocessing — brain mask, per-scan normalization, padding to 192×224, patient-wise split table
Stage A threshold baselines
Stage A U-Net training and evaluation (requires a GPU)
Stage B — ventricle masks, voxel table, models, SHAP, ablation, calibration, risk map

Requires: numpy, pandas, scipy, nibabel, matplotlib, torch, segmentation-models-pytorch, lightgbm, scikit-learn, shap. Stage A needs a GPU (~45 s/epoch on a Colab T4). Paths point at a Google Drive folder named Ms_project; change the PROJECT variable if yours differs.

Raw scans are not in this repository — download them from the Figshare link above and place them in data_raw/.

Use of AI assistance

AI tools were used to draft the code in this project. My own contribution was the data auditing, the design decisions (which scans to exclude, how to structure the split, which evaluation views to use), visual verification of every processing step, and the interpretation of results.

That verification was not a formality — it changed the project's conclusions three times:

The failed-brain-extraction explanation for the three outlier scans was proposed and then rejected by visual inspection, which showed clean brains on black backgrounds with no surrounding tissue.
The atrophy/enlarged-ventricle explanation that replaced it was rejected by quantification, since those scans had lower dark-voxel fractions than typical.
The hypothesis that centrally located lesions were disproportionately missed, prompted by seeing one untraced lesion in the centre of each worst-case scan, was rejected by testing it across all 1,932 held-out lesions, which showed no positional effect.

Several AI-suggested expectations also failed against the data and are reported as they came out rather than as predicted: that expert-mask features would outperform predicted-mask features, that the minimum-blob-size parameter was under-tuned at the edge of its grid, and that pooled AUPRC favoring logistic regression over LightGBM would hold up per patient.

Citation

Guarnera, F., Rondinella, A., Crispino, E., Russo, G., Di Lorenzo, C., Maimone, D., Pappalardo, F., & Battiato, S. (2025). MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset. Scientific Data, 12:920. https://doi.org/10.1038/s41597-025-05250-y
