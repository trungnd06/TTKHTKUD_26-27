## Slide 1: Single-trial EEG: frequent vs target (30 s)
Speaker: Lê Minh Đức

Hello. Our project classifies single EEG trials as frequent or target stimuli. We compare conventional machine learning with a compact convolutional network, then inspect which sensors and times affect the network. We use real data from the dataset specified in the course. This is an eight-subject pilot, so our conclusions are limited to this experiment.

## Slide 2: Three questions (40 s)
Speaker: Lê Minh Đức

We address three questions from the assignment. First, we rank the tested methods by mean held-out ROC-AUC. Second, we compare the CNN with classical methods on identical subjects and trials. Third, we investigate sensor and time sensitivity. Single-trial classification means one EEG response produces one prediction. The ERP averages shown later are visual summaries, not the classification inputs.

## Slide 3: Real data and preprocessing (55 s)
Speaker: Lê Minh Đức

We downloaded the oddball recordings of eight participants from OpenNeuro ds004621. S5 is the frequent stimulus and S6 is the target; we exclude S7 distractors and button-response markers. We selected sixteen sensors before evaluation. We average-reference those sensors, filter continuous signals from point five to thirty hertz, baseline-correct stimulus epochs, reject large or flat epochs, and resample to one hundred twenty-eight hertz. We did not apply ICA, so remaining artifacts are a limitation. The download manifest and preprocessing audit make these steps traceable.

## Slide 4: ERP signal and candidate features (55 s)
Speaker: Nguyễn Đức Trung

These are subject-weighted grand-average responses at frontal Fz and parietal Pz. Shading shows between-subject standard error. We compare the observed condition differences with the oddball ERP expectation. Averaged waveforms can look clearer than individual trials, which is why classification is harder than reading this plot. Classical models use twenty fifty-millisecond amplitude windows at each sensor. These features capture when a positive or negative response occurs without averaging different trials together.

## Slide 5: A comparison with held-out subjects (65 s)
Speaker: Nguyễn Đức Trung

Our outer split separates people, so test subjects never appear in training or validation within that fold. Each fold uses four people for training, two for validation and two for testing. Validation chooses the SVM penalty, forest depth and CNN checkpoint. The CNN uses temporal and grouped spatial convolutions inspired by EEGNet. We use class weighting for the rare target class. All methods share the same outer splits and retained trials. The test set is used only after fitting and selection. We report fold means and standard deviations, not confidence intervals.

## Slide 6: Which method performed best? (65 s)
Speaker: Phạm Sỹ Toàn

The highest mean ROC-AUC among the tested methods was 0.863, achieved by Compact EEG CNN. Its mean balanced accuracy was 0.775. The majority classifier illustrates why raw accuracy is insufficient: it favors the frequent class while its balanced accuracy and AUC remain one half. Our ranking is descriptive. It does not show a worldwide best method or guarantee performance on a new independent cohort.

## Slide 7: Did deep learning win? (50 s)
Speaker: Phạm Sỹ Toàn

The CNN had the higher mean held-out AUC in this pilot. The CNN-minus-strongest-classical-model mean AUC difference was +0.027. The ROC curves show the average sensitivity and false-positive tradeoff within held-out folds. This experiment cannot establish that deep learning is always better or worse. Our sample is small, training budgets differ, and we ran one base seed. A stronger follow-up would repeat seeds and validation splits and use the remaining participants. The report gives all fold-level results so variability remains visible.

## Slide 8: Where does the CNN look? (65 s)
Speaker: Đỗ Hoàng Minh

We use integrated gradients to trace the target logit from the training-mean baseline to each held-out trial. The largest mean absolute integrated-gradient channel totals were at Pz, P3, O2. The greatest mean AUC decrease under temporal occlusion occurred at 0.4-0.5 s (decrease 0.043). These are measured model-sensitivity summaries; they do not identify a neural generator. The map displays absolute sensitivity, so brighter values do not by themselves indicate positive target evidence. We compare these patterns with the ERP curves, while recognizing that eye movement, visual differences and motor responses may also contribute.

## Slide 9: Test the explanation by occlusion (60 s)
Speaker: Đỗ Hoàng Minh

We also replace one hundred milliseconds at a time with the training mean and measure the change in held-out AUC. The largest mean decrease occurred between 0.4 and 0.5 seconds. A positive bar indicates that removing that window worsened discrimination. Negative bars indicate improvement under this perturbation. Occlusion provides a second view of model reliance, although abrupt masking can create unnatural signals. Neither this analysis nor gradients prove a neural cause.

## Slide 10: Answers, limits and next steps (55 s)
Speaker: Đào Văn Việt

To conclude, Compact EEG CNN gave the highest mean AUC under our pilot protocol. The CNN had the higher mean held-out AUC in this pilot. We obtained partial sensor and time explanations through integrated gradients and occlusion. The main limits are the convenience sample, one seed and remaining artifacts or motor activity. We provide the code, exact splits, data hashes, predictions and a result-integrity checker. Our next steps are the full cohort, repeated seeds and negative controls. Thank you. We are ready for questions.

Total planned speaking time: 540 seconds (9 minutes).
