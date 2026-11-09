"""Generate English report, PowerPoint and Vietnamese study guide from measured results."""
import bootstrap
import csv
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches as DI, Pt as DP
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont, FontProperties

OUT = Path('deliverables')
RESULTS = Path('results')
MEMBERS = [('Lê Minh Đức', '24022291', 'Preprocessing and data quality'),
           ('Nguyễn Đức Trung', '24022473', 'ERP features and classical ML'),
           ('Phạm Sỹ Toàn', '24022467', 'CNN training and model comparison'),
           ('Đỗ Hoàng Minh', '24022399', 'Visualization and CNN explanations'),
           ('Đào Văn Việt', '24022489', 'Reproducibility, report and integration')]
REFERENCES = [
    '[1] Dzianok P et al. (2022). The Nencki-Symfonia electroencephalography/event-related potential dataset. GigaScience 11:giac015. https://doi.org/10.1093/gigascience/giac015',
    '[2] Dzianok P et al. Supporting data. GigaScience Database. https://doi.org/10.5524/100990',
    '[3] OpenNeuro ds004621, public dataset and metadata. https://github.com/OpenNeuroDatasets/ds004621 ; exact downloaded objects and SHA-256 hashes are in data/download_manifest.json.',
    '[4] Lawhern VJ et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces. https://arxiv.org/abs/1611.08024',
    '[5] Sundararajan M, Taly A, Yan Q (2017). Axiomatic Attribution for Deep Networks. https://arxiv.org/abs/1703.01365',
    '[6] MNE-Python, Epochs documentation. https://mne.tools/stable/generated/mne.Epochs.html',
    '[7] Course slides: neuroimaging_EEG_20261006.pdf, pp. 139-145, supplied with this project.'
]

def main():
    OUT.mkdir(exist_ok=True)
    m = json.loads((RESULTS / 'run_metadata.json').read_text())
    audit = json.loads(Path('data/preprocessing_audit.json').read_text())
    splits = json.loads((RESULTS / 'splits.json').read_text())
    df = pd.read_csv(RESULTS / 'fold_metrics.csv')
    avg = df.groupby('model').mean(numeric_only=True)
    std = df.groupby('model').std(numeric_only=True)
    ranked = avg.drop('Majority').sort_values('roc_auc', ascending=False)
    best = ranked.index[0]
    best_ml = avg.loc[['Shrinkage LDA', 'RBF SVM', 'Random Forest']].roc_auc.idxmax()
    cnn_auc, ml_auc = avg.loc['Compact EEG CNN', 'roc_auc'], avg.loc[best_ml, 'roc_auc']
    conclusion = ('The CNN had the higher mean held-out AUC in this pilot.' if cnn_auc > ml_auc else
                  f'{best_ml} had the higher mean held-out AUC in this pilot.')
    diff = cnn_auc - ml_auc
    before = sum(sum(a['before'].values()) for a in audit)
    dropped = sum(a['dropped'] for a in audit)
    imp = np.load(RESULTS / 'attributions.npz')
    spatial = imp['importance'].sum(1)
    top = [m['channels'][i] for i in np.argsort(spatial)[-3:][::-1]]
    occ = pd.read_csv(RESULTS / 'occlusion.csv').groupby('start_s').auc_drop.mean()
    peak = float(occ.idxmax())
    attr_text = (f'The largest mean absolute integrated-gradient channel totals were at {", ".join(top)}. '
                 f'The greatest mean AUC decrease under temporal occlusion occurred at {peak:.1f}-{peak + .1:.1f} s '
                 f'(decrease {occ.max():.3f}). These are measured model-sensitivity summaries; they do not identify a neural generator.')
    paired = df.pivot(index='fold', columns='model', values='roc_auc')
    differences = paired['Compact EEG CNN'] - paired[best_ml]
    table = [['Model', 'Balanced accuracy', 'ROC-AUC', 'PR-AUC (AP)', 'Target F1']]
    for name in ['Majority', 'Shrinkage LDA', 'RBF SVM', 'Random Forest', 'Compact EEG CNN']:
        table.append([name] + [f'{avg.loc[name, key]:.3f} ± {std.loc[name, key]:.3f}'
                              for key in ['balanced_accuracy', 'roc_auc', 'average_precision', 'f1']])
    sections = []
    def add(title, paragraphs, figure=None):
        sections.append((title, paragraphs, figure))
    add('Abstract', [
        f'We implemented a reproducible pilot for single-trial classification of frequent versus target EEG stimuli. '
        f'The experiment used {m["n_trials"]:,} retained trials from eight participants in the Nencki-Symfonia oddball dataset. '
        f'We compared shrinkage LDA, RBF SVM, Random Forest and a compact EEG CNN using four subject-disjoint outer folds. '
        f'The highest mean test ROC-AUC was {avg.loc[best, "roc_auc"]:.3f} ({best}); its balanced accuracy was '
        f'{avg.loc[best, "balanced_accuracy"]:.3f}. {conclusion} '
        'Integrated gradients and temporal occlusion supplied model explanations. '
        'These results concern a small, selected sample and one training seed; they do not establish a universal best classifier.'])
    add('1. Questions and scope', [
        'RQ1: Which tested model gives the best single-trial frequent-versus-target discrimination on unseen subjects? '
        'We define best primarily by mean outer-fold ROC-AUC, while reporting balanced accuracy and target-class F1 as complementary measures.',
        'RQ2: Does a deep model outperform conventional ML under the same trial and subject splits? '
        'RQ3: Which times and sensors influence the CNN, and are those patterns compatible with ERP physiology?',
        'A trial is the multichannel EEG response to one stimulus. ERP grand averages are used only for visualization. '
        'The classifiers receive individual trials; averaging trials before classification would answer a different question.'])
    add('2. Dataset and provenance', [
        'The course points to the Nencki-Symfonia dataset [1,2]. We downloaded its public OpenNeuro copy ds004621 [3]. '
        'The original study contains 42 adults and multiple tasks; this project uses only oddball. '
        'The task includes frequent standard stimuli, rare targets and rare distractors. We map S5 to standard (0), S6 to target (1), and omit S7 distractors and response markers.',
        'Our pilot subjects are ' + ', '.join(m['subjects']) + '. They were selected for a manageable local experiment. '
        'Sub-02 was skipped because the source paper reports a task-compliance problem; sub-14 and sub-17, also flagged there, are outside the pilot. '
        'This is a convenience subset, not a representative or randomly sampled cohort. No subject was excluded after inspecting classification performance.',
        f'There were {before:,} eligible standard/target events before rejection. We retained {m["n_standard"]:,} standards and '
        f'{m["n_target"]:,} targets; {dropped:,} events were dropped ({100*dropped/before:.1f}%). '
        'Each downloaded file has its source URL, byte count, ETag and SHA-256 in the manifest. This pins the actual downloaded bytes even though the public S3 path is not an immutable snapshot.'])
    add('3. Signal preprocessing', [
        'We used MNE-Python [6] to read BrainVision recordings. Sixteen sensors were fixed before evaluation: '
        + ', '.join(m['channels']) + '. This set covers frontal, central, parietal and occipital sites while reducing memory and model size.',
        'Continuous recordings were re-referenced to the average of these 16 sensors and filtered from 0.5 to 30 Hz with a fourth-order Butterworth IIR, using MNE forward-backward filtering. '
        'Epochs span -0.2 to 1.0 s relative to stimulus onset. Their pre-stimulus mean (-0.2 to 0 s) is subtracted. '
        'We reject an epoch if any retained channel has peak-to-peak amplitude above 150 microvolts or below 0.1 microvolts, then resample to 128 Hz. '
        'Distractors and button-response events never become classification examples.',
        'These operations use fixed parameters and no class-fitted transforms. Each subject is processed independently. '
        'No ICA or manual component rejection was performed; residual ocular and muscle activity may remain. '
        'The selected-sensor reference differs from a full-cap average reference. A 30 Hz low-pass already attenuates 50 Hz line noise. '
        'Zero-phase filtering uses future samples, so this implementation is an offline analysis, not a validated online BCI.'], 'erp.png')
    add('4. Models and training', [
        'Classical features are 20 nonoverlapping 50 ms mean-amplitude windows between 0 and 1 s at each sensor, giving 320 features per trial. '
        'They preserve ERP timing and polarity. Standardization is fitted on training trials only for LDA and SVM. '
        'The majority classifier reports training class priors and predicts the frequent class at threshold 0.5.',
        'Shrinkage LDA uses automatic covariance shrinkage with equal class priors. '
        'RBF SVM uses balanced class weights and chooses C from {0.1, 1, 10} by validation AUC; gamma uses scikit-learn scale. '
        'Its probability calibration is internal to training data. Random Forest uses 200 trees, minimum leaf size 3, balanced class weights, and validation chooses depth 6 or unlimited.',
        'Our compact CNN is inspired by EEGNet depthwise/separable convolutions [4], but is not an exact reproduction. '
        'It receives the full baseline-corrected epoch, including the pre-stimulus interval. '
        'Eight 33-sample temporal filters precede 16 grouped spatial filters, batch normalization, ELU, average pooling and dropout 0.35. '
        'A depthwise 9-sample temporal convolution and pointwise convolution precede adaptive pooling to 8 time bins and a linear target logit.',
        'CNN normalization uses each channel mean and standard deviation over training trials and time. '
        'Training uses AdamW (learning rate 0.001, weight decay 0.01), batch size 64, gradient clipping at norm 5, '
        f'a maximum of {m["max_epochs"]} epochs, and patience 8 on validation AUC. '
        'Binary cross-entropy weights the target class by the training standard-to-target ratio. The best validation checkpoint is restored. '
        f'The base random seed is {m["seed"]}; CNN fold seeds are base + fold.'])
    add('5. Evaluation and safeguards', [
        'Four GroupKFold outer splits hold out two subjects each. In each outer development set, two other subjects form validation and the remaining four form training. '
        'Validation membership follows a fixed rotation of sorted development subject IDs, without using labels or performance. '
        'Exact memberships are recorded below and in results/splits.json. Every retained trial is tested exactly once per model.',
        'Only validation AUC selects hyperparameters and CNN epochs. Test labels are used only for evaluation and post-training explanation. '
        'All models use identical trial splits. No oversampling, test normalization fitting or test threshold tuning is performed. '
        'The classification threshold is fixed at probability 0.5. CNN weighted-loss scores are not guaranteed to be calibrated probabilities.',
        'ROC-AUC measures score ranking; balanced accuracy is the mean of target sensitivity and standard specificity. '
        'Average precision summarizes the precision-recall curve and is sensitive to prevalence. '
        'We report mean +/- sample SD across four folds. These folds share training subjects, so their SD is descriptive and is not a confidence interval. '
        'The best observed mean is a descriptive ranking among the tested methods; it is not an unbiased estimate after model-family selection.'], 'comparison.png')
    add('6. Measured classification results', [
        f'{best} had the highest mean held-out ROC-AUC ({avg.loc[best, "roc_auc"]:.3f} +/- {std.loc[best, "roc_auc"]:.3f}). '
        f'Its balanced accuracy was {avg.loc[best, "balanced_accuracy"]:.3f} +/- {std.loc[best, "balanced_accuracy"]:.3f}. '
        f'The CNN AUC was {cnn_auc:.3f}; the strongest classical model by mean AUC was {best_ml} ({ml_auc:.3f}). '
        f'The paired foldwise CNN-minus-ML differences were ' + ', '.join(f'{d:+.3f}' for d in differences) + '.',
        f'{conclusion} The mean difference was {diff:+.3f} AUC. '
        'This is not evidence of universal superiority: the subset is small, budgets differ by model family, and there are no repeated training seeds or independent final cohort.',
        f'Target prevalence after rejection was {m["n_target"]/m["n_trials"]:.1%}. '
        f'The majority baseline mean accuracy was {avg.loc["Majority", "accuracy"]:.3f}, '
        'despite balanced accuracy and ROC-AUC both being 0.500. This shows why raw accuracy alone is misleading. '
        'The pooled confusion matrices below use only out-of-fold predictions. ROC curves average within-fold curves rather than treating independently calibrated fold scores as one common scale.'], 'confusion.png')
    add('7. Explaining the deep model', [
        'Integrated gradients [5] integrate the gradient of the target logit along a straight path from the training channel-mean baseline to the trial. '
        'We use 24 midpoint integration steps and aggregate absolute attribution over equal numbers of standard and target held-out trials, at most 32 per class per fold. '
        'The heatmap averages these per-fold maps equally. Absolute values show sensitivity magnitude and discard direction; they should not be read as signed evidence for target.',
        attr_text,
        'For temporal occlusion, each 100 ms post-stimulus window is replaced by the training channel mean in standardized space, '
        'and held-out ROC-AUC is recalculated. A positive decrease means that removing this window worsened ranking. '
        'A negative value means performance improved under that perturbation. Sudden window boundaries can create artificial input patterns.',
        'In our Pz ERP plot, targets are more positive than standards around 0.3-0.6 s. The Pz attribution hotspot around 0.4-0.6 s and the 0.4-0.5 s occlusion maximum '
        'are compatible with a P300-related condition difference. Absolute amplitudes and polarity depend on our 16-sensor reference, which differs from the source study. '
        'Targets also require a motor response, so models may exploit response-related signals, visual differences, or residual artifacts. '
        'Scalp attribution is neither source localization nor causal proof of cognitive mechanisms. We do not claim fully explainable or purely P300-based decisions.'], 'attribution.png')
    add('8. Limitations and next experiments', [
        'The main limits are eight convenience-selected participants, one base seed, one fixed validation split per outer fold, '
        'a limited hyperparameter search, 16 selected sensors and residual artifacts. '
        'The CNN sees a short pre-stimulus interval whereas classical ERP features start at onset; this is a documented input difference. '
        'Post-stimulus motor activity can confound the interpretation of attention-related classification.',
        'Useful extensions are the remaining cohort, repeated subject-disjoint validation, multiple CNN seeds, '
        'a matched post-stimulus-only CNN, and a pre-stimulus-only negative control. '
        'Artifact sensitivity, pre-response-only windows and comparisons with distractors could separate alternative explanations. '
        'Such experiments are proposed future work and are not represented as completed results.'], 'occlusion.png')
    add('9. Answers to the course questions', [
        f'RQ1: Within our tested models and pilot protocol, {best} has the highest mean unseen-subject ROC-AUC, '
        f'{avg.loc[best, "roc_auc"]:.3f}. This is a pilot result, not a global optimum.',
        f'RQ2: {conclusion} CNN-minus-best-ML mean AUC = {diff:+.3f}. '
        'The experiment supports a comparison under the stated protocol, not a general claim that one learning paradigm is always better.',
        'RQ3: We can inspect sensor/time sensitivity using integrated gradients and occlusion, and compare it with ERP patterns. '
        'These explanations are partial and baseline-dependent; they do not prove which physiological source caused the prediction.'])
    add('10. Reproducibility and verification', [
        'Run download_data.py, preprocess.py, experiment.py, verify_results.py and build_deliverables.py, in that order. '
        'All scripts run from the project directory. Models, split memberships, per-trial predictions, fold metrics, learning curves and attribution arrays are saved.',
        'verify_results.py checks disjoint subjects, complete out-of-fold coverage, label consistency, metric reconstruction, the majority baseline, '
        'and numerical integrated-gradient completeness on actual held-out examples. '
        'Exact library versions and seeds are recorded in results/run_metadata.json. Different library or hardware versions may give slightly different numerical results.',
        f'Environment: Python {m["python"]}, NumPy {m["numpy"]}, SciPy {m["scipy"]}, '
        f'scikit-learn {m["sklearn"]}, PyTorch {m["torch"]}.'])
    add('11. Group contributions and submission', [
        'The following is a proposed equal work allocation, not a verified statement of completed individual work. '
        'Each member must review and reproduce their part, record their actual changes and understanding, '
        'then confirm or revise the percentages before submission. Proposed percentages total 100%. '
        'Automation-assisted project preparation does not establish any individual contribution.',
        'The English presentation is designed for nine minutes, leaving one minute below the ten-minute maximum. '
        'All five members must attend and prepare for individual questions. '
        'Send the English report at least one week before presentation to guy.nagels@vub.be, Mr Tran Quoc Sang, and Mr Nguyen Tien Dung, '
        'using the latter two recipients\' addresses supplied by the course. Presentation date and course-specific cover details have not been supplied.'])
    add('References', REFERENCES)
    # Shared content drives Markdown, DOCX and PDF to keep results consistent.
    title = 'Single-Trial EEG Classification of Frequent and Target Stimuli'
    subtitle = 'Nencki-Symfonia dataset | A reproducible eight-subject pilot'
    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = DP(11)
    doc.add_heading(title, 0); doc.add_paragraph(subtitle)
    for name, sid, _ in MEMBERS:
        doc.add_paragraph(f'{name} - {sid}')
    doc.add_paragraph('Prepared for the miniproject requirements on course slides 139-145.')
    doc.add_paragraph('Contribution allocation requires group confirmation before submission.')
    doc.add_page_break()
    md = [f'# {title}', subtitle, '\n'.join(f'- {n} — {s}' for n, s, _ in MEMBERS)]
    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(TTFont('ProjectSans', findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('ProjectSansBold', findfont(FontProperties(family='DejaVu Sans', weight='bold'))))
    for style in ['Normal', 'BodyText', 'Title', 'Heading1', 'Heading2']:
        styles[style].fontName = 'ProjectSansBold' if style in ['Title', 'Heading1', 'Heading2'] else 'ProjectSans'
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].keepWithNext = True
    styles['BodyText'].fontSize = 10; styles['BodyText'].leading = 14
    pdf = [Paragraph(html.escape(title), styles['Title']), Paragraph(subtitle, styles['Heading2']), Spacer(1, 16)]
    for n, sid, _ in MEMBERS:
        pdf.append(Paragraph(f'{n} - {sid}', styles['BodyText']))
    pdf += [Spacer(1, 24), Paragraph('Contribution allocation requires group confirmation before submission.', styles['BodyText']), PageBreak()]
    for heading, paras, figure in sections:
        if heading == 'References':
            doc.add_page_break()
            pdf.append(PageBreak())
        doc.add_heading(heading, 1)
        md += [f'## {heading}'] + paras
        pdf.append(Paragraph(html.escape(heading), styles['Heading1']))
        for para in paras:
            doc.add_paragraph(para)
            pdf += [Paragraph(html.escape(para), styles['BodyText']), Spacer(1, 7)]
        if heading.startswith('6.'):
            add_doc_table(doc, table)
            md += ['| ' + ' | '.join(table[0]) + ' |', '| ' + ' | '.join(['---'] * len(table[0])) + ' |']
            md += ['| ' + ' | '.join(row) + ' |' for row in table[1:]]
            pdf.append(pdf_table(table, [108, 93, 90, 90, 90], styles))
            pdf.append(Spacer(1, 12))
        if heading.startswith('2.'):
            counts = [['Subject', 'Standard kept', 'Target kept', 'Dropped']] + [
                [a['subject'], str(a['after']['standard']), str(a['after']['target']), str(a['dropped'])] for a in audit]
            add_doc_table(doc, counts); pdf += [pdf_table(counts, [117] * 4, styles), Spacer(1, 12)]
        if heading.startswith('5.'):
            membership = [['Fold', 'Train', 'Validation', 'Test']] + [
                [str(s['fold'])] + [', '.join(x.replace('sub-', '') for x in s[k]) for k in ['train', 'validation', 'test']] for s in splits]
            add_doc_table(doc, membership); pdf += [pdf_table(membership, [35, 190, 122, 122], styles), Spacer(1, 12)]
        if heading.startswith('11.'):
            contrib = [['Member / ID', 'Proposed responsibility', 'Proposed %']] + [
                [f'{n} / {sid}', role, '20'] for n, sid, role in MEMBERS]
            add_doc_table(doc, contrib)
            pdf += [pdf_table(contrib, [155, 260, 56], styles), Spacer(1, 12)]
        if figure:
            doc.add_picture(str(RESULTS / figure), width=DI(6.25))
            md.append(f'![{figure}](../results/{figure})')
            from PIL import Image as PILImage
            with PILImage.open(RESULTS / figure) as im:
                width, height = im.size
            pdf += [Image(str(RESULTS / figure), width=6.5*inch, height=6.5*inch*height/width), Spacer(1, 12)]
    doc.save(OUT / 'Report_EN.docx')
    (OUT / 'Report_EN.md').write_text('\n\n'.join(md), encoding='utf-8')
    SimpleDocTemplate(str(OUT / 'Report_EN.pdf'), rightMargin=54, leftMargin=54,
                      topMargin=48, bottomMargin=48).build(pdf, onFirstPage=page_number, onLaterPages=page_number)
    with (OUT / 'contributions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'student_id', 'proposed_role', 'proposed_percent', 'confirmed', 'actual_evidence'])
        for n, sid, role in MEMBERS:
            writer.writerow([n, sid, role, 20, 'NO', 'To be completed by the member'])
    slides = build_slides(m, avg, std, best, best_ml, conclusion, diff, attr_text, peak)
    make_pptx(slides)
    (OUT / 'Presentation_script_EN.md').write_text('\n\n'.join(
        f'## Slide {i}: {s[0]} ({s[4]} s)\nSpeaker: {s[5]}\n\n{s[3]}' for i, s in enumerate(slides, 1)) +
        '\n\nTotal planned speaking time: 540 seconds (9 minutes).\n', encoding='utf-8')
    write_guide(best, best_ml, conclusion, diff)
    (OUT / 'requirements_checklist.md').write_text('''# Course requirements, pp. 139-145

| Requirement | Evidence |
|---|---|
| Nencki-Symfonia oddball data | data/download_manifest.json; report section 2 |
| Frequent versus target, single trial | S5/S6 labels; experiment.py; per-trial predictions |
| Best result among tested methods | report section 6; results/summary.csv |
| Deep learning versus ML | LDA, SVM, RF, CNN; same outer splits |
| Explain deep features | integrated gradients and occlusion; report section 7 |
| Report in English | Report_EN.docx / PDF |
| Individual roles and percentages sum to 100% | contributions.csv: 5 x 20%, proposed; confirmation pending |
| English presentation <=10 minutes | Presentation_EN.pptx; 540-second script |
| Submit >=1 week before presentation | Date not supplied; group must schedule and send |
| All members attend and answer questions | Vietnamese study guide; group action required |
''', encoding='utf-8')
    print('Generated report, presentation, script, guide and checklist in deliverables/')

def add_doc_table(doc, rows):
    t = doc.add_table(rows=1, cols=len(rows[0])); t.style = 'Light Shading Accent 1'
    for cell, text in zip(t.rows[0].cells, rows[0]): cell.text = text
    for row in rows[1:]:
        for cell, text in zip(t.add_row().cells, row): cell.text = text

def pdf_table(rows, widths, styles):
    from reportlab.lib.styles import ParagraphStyle
    cell_style = ParagraphStyle('TableCell', parent=styles['BodyText'], fontSize=9, leading=12)
    t = Table([[Paragraph(html.escape(str(c)), cell_style) for c in row] for row in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dce8f3')),
                           ('GRID', (0,0), (-1,-1), .4, colors.HexColor('#cbd5e1')),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING', (0,0), (-1,-1), 6)]))
    return KeepTogether([t])

def page_number(canvas, doc):
    canvas.setFont('Helvetica', 9); canvas.drawRightString(540, 28, str(doc.page))

def build_slides(m, avg, std, best, best_ml, conclusion, diff, attr_text, peak):
    # title, bullets, figure, notes, seconds, speaker
    return [
        ('Single-trial EEG: frequent vs target',
         ['Nencki-Symfonia | eight-subject pilot', 'Classical ML, compact CNN and model explanations',
          'Lê Minh Đức · Nguyễn Đức Trung · Phạm Sỹ Toàn', 'Đỗ Hoàng Minh · Đào Văn Việt'], None,
         'Hello. Our project classifies single EEG trials as frequent or target stimuli. We compare conventional machine learning with a compact convolutional network, then inspect which sensors and times affect the network. We use real data from the dataset specified in the course. This is an eight-subject pilot, so our conclusions are limited to this experiment.', 30, MEMBERS[0][0]),
        ('Three questions', ['Which tested method performs best on unseen people?',
          'Does deep learning outperform classical ML here?', 'Can we interpret what the CNN uses?'], None,
         'We address three questions from the assignment. First, we rank the tested methods by mean held-out ROC-AUC. Second, we compare the CNN with classical methods on identical subjects and trials. Third, we investigate sensor and time sensitivity. Single-trial classification means one EEG response produces one prediction. The ERP averages shown later are visual summaries, not the classification inputs.', 40, MEMBERS[0][0]),
        ('Real data and preprocessing',
         [f'{m["n_trials"]:,} clean trials: {m["n_standard"]:,} standard / {m["n_target"]:,} target',
          'S5 = standard; S6 = target; distractors excluded', '16 sensors · 0.5–30 Hz · 128 Hz',
          'Epoch −0.2 to 1.0 s · baseline correction · amplitude rejection'], None,
         'We downloaded the oddball recordings of eight participants from OpenNeuro ds004621. S5 is the frequent stimulus and S6 is the target; we exclude S7 distractors and button-response markers. We selected sixteen sensors before evaluation. We average-reference those sensors, filter continuous signals from point five to thirty hertz, baseline-correct stimulus epochs, reject large or flat epochs, and resample to one hundred twenty-eight hertz. We did not apply ICA, so remaining artifacts are a limitation. The download manifest and preprocessing audit make these steps traceable.', 55, MEMBERS[0][0]),
        ('ERP signal and candidate features',
         ['Subject-weighted mean; shading = between-subject SE', 'ML: 20 × 50-ms means per sensor = 320 features'], 'erp.png',
         'These are subject-weighted grand-average responses at frontal Fz and parietal Pz. Shading shows between-subject standard error. We compare the observed condition differences with the oddball ERP expectation. Averaged waveforms can look clearer than individual trials, which is why classification is harder than reading this plot. Classical models use twenty fifty-millisecond amplitude windows at each sensor. These features capture when a positive or negative response occurs without averaging different trials together.', 55, MEMBERS[1][0]),
        ('A comparison with held-out subjects',
         ['4 outer folds: 2 test subjects each', 'Within each fold: 4 train + 2 validation + 2 test',
          'Train-only scaling; validation chooses settings', 'LDA · RBF SVM · Random Forest · compact EEG CNN'], None,
         'Our outer split separates people, so test subjects never appear in training or validation within that fold. Each fold uses four people for training, two for validation and two for testing. Validation chooses the SVM penalty, forest depth and CNN checkpoint. The CNN uses temporal and grouped spatial convolutions inspired by EEGNet. We use class weighting for the rare target class. All methods share the same outer splits and retained trials. The test set is used only after fitting and selection. We report fold means and standard deviations, not confidence intervals.', 65, MEMBERS[1][0]),
        ('Which method performed best?',
         [f'Best mean AUC: {best} = {avg.loc[best,"roc_auc"]:.3f} ± {std.loc[best,"roc_auc"]:.3f}',
          f'Balanced accuracy: {avg.loc[best,"balanced_accuracy"]:.3f}',
          'Error bars = fold SD; majority baseline = 0.5'], 'comparison.png',
         f'The highest mean ROC-AUC among the tested methods was {avg.loc[best,"roc_auc"]:.3f}, achieved by {best}. '
         f'Its mean balanced accuracy was {avg.loc[best,"balanced_accuracy"]:.3f}. '
         'The majority classifier illustrates why raw accuracy is insufficient: it favors the frequent class while its balanced accuracy and AUC remain one half. '
         'Our ranking is descriptive. It does not show a worldwide best method or guarantee performance on a new independent cohort.', 65, MEMBERS[2][0]),
        ('Did deep learning win?',
         [conclusion, f'CNN − strongest ML: {diff:+.3f} mean AUC',
          'One seed · eight participants · limited search', 'More data and repeated seeds are needed'], 'roc.png',
         f'{conclusion} The CNN-minus-strongest-classical-model mean AUC difference was {diff:+.3f}. '
         'The ROC curves show the average sensitivity and false-positive tradeoff within held-out folds. '
         'This experiment cannot establish that deep learning is always better or worse. Our sample is small, training budgets differ, and we ran one base seed. '
         'A stronger follow-up would repeat seeds and validation splits and use the remaining participants. The report gives all fold-level results so variability remains visible.', 50, MEMBERS[2][0]),
        ('Where does the CNN look?',
         ['Integrated gradients on held-out trials', 'Training-mean baseline; absolute sensitivity',
          'Sensor/time evidence is not causal localization'], 'attribution.png',
         'We use integrated gradients to trace the target logit from the training-mean baseline to each held-out trial. '
         + attr_text + ' The map displays absolute sensitivity, so brighter values do not by themselves indicate positive target evidence. '
         'We compare these patterns with the ERP curves, while recognizing that eye movement, visual differences and motor responses may also contribute.', 65, MEMBERS[3][0]),
        ('Test the explanation by occlusion',
         [f'Largest mean AUC drop: {peak:.1f}–{peak+.1:.1f} s',
          'Replace one 100-ms window with training mean', 'Perturbation evidence; motor/artifact confounds remain'], 'occlusion.png',
         'We also replace one hundred milliseconds at a time with the training mean and measure the change in held-out AUC. '
         f'The largest mean decrease occurred between {peak:.1f} and {peak+.1:.1f} seconds. '
         'A positive bar indicates that removing that window worsened discrimination. Negative bars indicate improvement under this perturbation. '
         'Occlusion provides a second view of model reliance, although abrupt masking can create unnatural signals. Neither this analysis nor gradients prove a neural cause.', 60, MEMBERS[3][0]),
        ('Answers, limits and next steps',
         [f'RQ1: {best} leads the tested pilot by AUC', f'RQ2: CNN − strongest ML = {diff:+.3f}',
          'RQ3: partial explanations from gradients + occlusion',
          'Next: full cohort, repeated seeds, artifact and pre-stimulus controls'], None,
         f'To conclude, {best} gave the highest mean AUC under our pilot protocol. '
         f'{conclusion} We obtained partial sensor and time explanations through integrated gradients and occlusion. '
         'The main limits are the convenience sample, one seed and remaining artifacts or motor activity. '
         'We provide the code, exact splits, data hashes, predictions and a result-integrity checker. '
         'Our next steps are the full cohort, repeated seeds and negative controls. Thank you. We are ready for questions.', 55, MEMBERS[4][0])]

def make_pptx(slides):
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    navy = RGBColor.from_string('10243A'); orange = RGBColor.from_string('DC7633')
    def textbox(slide, x, y, w, h, text, size=22, color=navy, bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame; tf.word_wrap = True
        for j, line in enumerate(text.split('\n')):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(size); p.font.color.rgb = color; p.font.bold = bold
            p.space_after = Pt(14)
        return box
    for i, (title, bullets, figure, notes, seconds, speaker) in enumerate(slides, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor.from_string('F8FAFC')
        textbox(slide, .55, .35, 12.2, .7, title, 30, bold=True)
        if figure:
            textbox(slide, .6, 1.22, 12, .85, '  |  '.join(bullets), 17)
            from PIL import Image as PILImage
            with PILImage.open(RESULTS / figure) as im: w, h = im.size
            pw, ph = 11.7, 4.65
            if pw / ph > w / h: pw = ph * w / h
            else: ph = pw * h / w
            slide.shapes.add_picture(str(RESULTS / figure), Inches((13.333-pw)/2), Inches(2.02), width=Inches(pw), height=Inches(ph))
        else:
            textbox(slide, .8, 1.65, 11.7, 4.9, '\n'.join('• ' + b for b in bullets), 25)
        textbox(slide, .6, 7.04, 11.5, .3, f'Nencki-Symfonia EEG pilot  •  {speaker}  •  {seconds} s', 11)
        textbox(slide, 12.15, 7.0, .6, .3, f'{i:02d}', 12, orange)
        slide.notes_slide.notes_text_frame.text = notes
    # Two backup slides carry sources and the explicitly provisional allocation.
    for title, text in [('References (backup)', '\n'.join(REFERENCES[:6])),
                        ('Proposed contributions (backup)', '\n'.join(f'{n} | {sid} | {role} | 20%' for n,sid,role in MEMBERS) +
                         '\nTotal 100%. Proposed only: confirm actual individual work before submission.')]:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        textbox(slide, .6, .4, 12, .7, title, 30, bold=True)
        textbox(slide, .65, 1.45, 12, 5.7, text, 18)
    prs.save(OUT / 'Presentation_EN.pptx')

def write_guide(best, best_ml, conclusion, diff):
    text = f'''# Hướng dẫn hoàn thiện và bảo vệ miniproject

## Những gì đã thực hiện

Dữ liệu thật: oddball, 8 người của Nencki–Symfonia, bản OpenNeuro ds004621.
Đã chạy tiền xử lý, majority baseline, shrinkage LDA, RBF SVM, Random Forest, CNN; đánh giá 4 fold theo người.
Kết quả chính: {best} có ROC-AUC trung bình cao nhất trong các mô hình đã thử.
CNN trừ ML mạnh nhất ({best_ml}) = {diff:+.3f} AUC. Xem số chi tiết trong Report_EN.pdf và results/summary.csv.
Kết quả là pilot 8 người, không phải toàn bộ 42 người hay kỷ lục của bộ dữ liệu.

## Mỗi thành viên cần tự làm và hiểu

Bảng dưới là **đề xuất**, không xác nhận ai đã thực hiện công việc. Mỗi người phải kiểm tra, chạy lại,
ghi nhận phần việc thực tế và điều chỉnh tỷ lệ trong contributions.csv (tổng 100%).
Việc được giao tên trong tài liệu không tự chứng minh đóng góp.

| Thành viên | Phần đề xuất | Minh chứng nên chuẩn bị |
|---|---|---|
| Lê Minh Đức — 24022291 | Dữ liệu, tiền xử lý | Giải thích S5/S6/S7, chạy preprocess.py, đọc audit |
| Nguyễn Đức Trung — 24022473 | Đặc trưng và ML | Giải thích 320 đặc trưng, LDA/SVM/RF, train-only scaling |
| Phạm Sỹ Toàn — 24022467 | CNN và so sánh | Vẽ lại kiến trúc, giải thích loss và validation checkpoint |
| Đỗ Hoàng Minh — 24022399 | Hình và giải thích | Đọc attribution/occlusion, nêu hạn chế suy luận |
| Đào Văn Việt — 24022489 | Tích hợp, báo cáo, tái lập | Chạy verify_results.py, tổng hợp báo cáo, kiểm tra nguồn |

Tạm đề xuất mỗi người 20%; nhóm cần xác nhận hoặc sửa theo thực tế. Có thể đổi người trình bày trong slide.

## Câu hỏi vấn đáp

1. **Single trial là gì?** Một đáp ứng EEG sau một kích thích. Mỗi trial sinh một dự đoán. ERP là trung bình nhiều trial để nhìn dạng sóng.
2. **Nhãn lấy ở đâu?** Marker S5 = thường gặp, S6 = mục tiêu; S7 = distractor bị loại. Không dùng marker bấm nút làm nhãn.
3. **Vì sao không chỉ dùng accuracy?** Lớp standard chiếm đa số. Đoán toàn standard vẫn có accuracy cao nhưng balanced accuracy và AUC chỉ 0.5.
4. **Balanced accuracy tính sao?** (Sensitivity của target + specificity của standard) / 2.
5. **ROC-AUC là gì?** Khả năng xếp điểm target cao hơn standard khi thay đổi ngưỡng. Không phải tỷ lệ dự đoán đúng ở ngưỡng 0.5.
6. **Có rò rỉ dữ liệu không?** Trong từng fold, train/validation/test là các người khác nhau. Scaler chỉ fit train; không dùng test chọn C hay epoch.
7. **Vì sao chia theo người?** Trial của cùng một người có đặc điểm tương quan. Trộn trial vào train/test sẽ không đo đúng khái quát sang người mới.
8. **Vì sao có validation?** Chọn cấu hình mà không nhìn điểm test. 2 người test, 2 validation, 4 train ở mỗi fold.
9. **Đặc trưng ML là gì?** Trung bình biên độ từng cửa sổ 50 ms trong 0–1 s, 20 cửa sổ × 16 kênh = 320.
10. **CNN học gì?** Lọc theo thời gian rồi phối hợp không gian giữa điện cực; lớp cuối tạo logit target. Kiến trúc lấy ý tưởng EEGNet, không sao chép chính xác EEGNet.
11. **Vì sao weighting?** Target hiếm; trọng số trong loss tăng chi phí bỏ sót target. Tính tỷ lệ từ train, không từ test.
12. **Deep learning có tốt hơn không?** Trả lời theo bảng kết quả: {conclusion} Không khái quát sang mọi dữ liệu từ một pilot.
13. **Integrated gradients là gì?** Tích phân gradient logit dọc đường từ baseline đến input. Baseline là trung bình kênh của train; đồ thị lấy trị tuyệt đối nên không cho chiều tác động.
14. **Occlusion là gì?** Che một cửa sổ 100 ms bằng trung bình train, đo AUC giảm bao nhiêu. Che có thể tạo tín hiệu không tự nhiên.
15. **Có chứng minh P300 không?** Không. Có thể so với ERP nhưng còn nhiễu, khác biệt thị giác và vận động bấm nút. Attribution trên da đầu không định vị nguồn não.
16. **Đã loại nhiễu mắt bằng ICA chưa?** Chưa. Có lọc, tham chiếu, baseline và loại epoch theo biên độ. Báo cáo ghi rõ giới hạn này.
17. **Tại sao không chạy 42 người?** Chọn 8 người để phù hợp tài nguyên/thời gian miniproject. Có mã mở rộng, nhưng điểm báo cáo chỉ thuộc 8 người đã chạy.
18. **Mean ± SD có phải khoảng tin cậy không?** Không. SD cho độ dao động qua 4 fold; các fold còn dùng chung một số người trong train.
19. **Dùng online được chưa?** Chưa. Lọc hai chiều dùng mẫu tương lai, và chưa đo độ trễ hay thử hệ thống online.
20. **Cải tiến tiếp theo?** Toàn bộ cohort, nhiều seed, lặp validation, kiểm soát chỉ trước kích thích và trước phản hồi, kiểm tra nhạy cảm với nhiễu.

## Trình bày và nộp

- 10 slide chính có kịch bản tổng 540 giây; 2 slide dự phòng về nguồn và đóng góp. Luyện nói và đo giờ thực tế.
- Mọi người cần hiểu toàn bộ pipeline, không chỉ phần được giao. Slide gốc cho biết điểm cá nhân phụ thuộc đóng góp và hiểu biết.
- Gửi báo cáo tiếng Anh ít nhất 1 tuần trước lịch thuyết trình cho guy.nagels@vub.be và hai thầy Trần Quốc Sang, Nguyễn Tiến Dũng (lấy email từ lớp).
- Chưa có ngày thuyết trình, tên lớp và mẫu bìa riêng. Bổ sung nếu giảng viên yêu cầu.
- Mở Word/PDF/PPTX, kiểm tra tên, tỷ lệ, biểu đồ; không nộp phần phân công đề xuất như đóng góp đã xác nhận.
- Công cụ không tự gửi email cho giảng viên.
'''
    (OUT / 'HUONG_DAN_VA_VAN_DAP_VI.md').write_text(text, encoding='utf-8')

if __name__ == '__main__':
    main()
