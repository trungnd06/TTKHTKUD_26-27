"""Build the full-cohort report and slides from the actual Kaggle outputs."""
import bootstrap
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import build_deliverables as layout
from docx import Document
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image as PILImage

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results_full'))
    p.add_argument('--output', type=Path, default=Path('deliverables_full'))
    args = p.parse_args()
    source, out = args.results, args.output
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / 'run_metadata.json').read_text(encoding='utf-8'))
    prep = json.loads((source / 'preparation_summary.json').read_text(encoding='utf-8'))
    assert meta['data_sha256'] == prep['processed_sha256']
    assert meta['subjects'] == prep['included_subjects']
    assert len(meta['subjects']) == 42 and prep['all_42_attempted']
    for k in ['n_trials', 'n_standard', 'n_target']:
        assert meta[k] == prep[k]
    assert json.loads((source / 'verification.json').read_text())['passed']
    df = pd.read_csv(source / 'fold_metrics.csv')
    splits = json.loads((source / 'splits.json').read_text())
    avg, sd = df.groupby('model').mean(numeric_only=True), df.groupby('model').std(numeric_only=True)
    best = avg.drop('Majority').roc_auc.idxmax()
    ml = avg.loc[['Shrinkage LDA', 'RBF SVM', 'Random Forest']].roc_auc.idxmax()
    delta = float(avg.loc['Compact EEG CNN', 'roc_auc'] - avg.loc[ml, 'roc_auc'])
    differences = df.pivot(index='fold', columns='model', values='roc_auc')
    differences = differences['Compact EEG CNN'] - differences[ml]
    before = sum(sum(a['before'].values()) for a in prep['audits'])
    dropped = sum(a['dropped'] for a in prep['audits'])
    boundaries = [a['subject'] for a in prep['audits'] if a['interior_boundaries_s']]
    occ = pd.read_csv(source / 'occlusion.csv').groupby('start_s').auc_drop.agg(['mean', 'std'])
    peak = float(occ['mean'].idxmax())
    attrs = np.load(source / 'attributions.npz')
    top = [str(attrs['channels'][i]) for i in np.argsort(attrs['importance'].sum(1))[-3:][::-1]]
    attribution = (f'The three largest mean absolute channel-attribution totals are {", ".join(top)}. '
                   f'The greatest mean occlusion AUC decrease is {occ.loc[peak,"mean"]:.3f} '
                   f'in the {peak:.1f}-{peak+.1:.1f} s window. These quantities describe model reliance, not a causal neural source.')
    conclusion = f'The CNN exceeds {ml} by {delta:.3f} mean ROC-AUC in this experiment.'
    refs = list(layout.REFERENCES)
    refs[2] = ('[3] OpenNeuro ds004621. https://github.com/OpenNeuroDatasets/ds004621 . '
               'Downloaded file URLs, ETags and SHA-256 values are recorded per subject in results_full/preparation_summary.json.')
    sections = []
    def add(title, paragraphs, figure=None, table=None):
        sections.append((title, paragraphs, figure, table))
    score_table = [['Model', 'ROC-AUC', 'Balanced accuracy', 'Target F1', 'Accuracy']]
    for name in ['Majority', 'Shrinkage LDA', 'RBF SVM', 'Random Forest', 'Compact EEG CNN']:
        score_table.append([name] + [f'{avg.loc[name,k]:.3f} ± {sd.loc[name,k]:.3f}'
                           for k in ['roc_auc','balanced_accuracy','f1','accuracy']])
    add('Abstract', [
        f'We classify single EEG trials as frequent or target stimuli in the Nencki-Symfonia oddball task. '
        f'The full available oddball cohort comprises 42 participants. After fixed signal-quality rejection, '
        f'{meta["n_trials"]:,} trials remain: {meta["n_standard"]:,} standards and {meta["n_target"]:,} targets. '
        'We compare shrinkage linear discriminant analysis, RBF support vector machines, Random Forest and a compact EEG convolutional network. '
        f'Five subject-disjoint outer folds yield a CNN ROC-AUC of {avg.loc[best,"roc_auc"]:.3f} ± {sd.loc[best,"roc_auc"]:.3f} '
        f'and balanced accuracy of {avg.loc[best,"balanced_accuracy"]:.3f} ± {sd.loc[best,"balanced_accuracy"]:.3f}. '
        f'{conclusion} Integrated gradients and temporal occlusion indicate sensor/time reliance compatible with ERP condition differences. '
        'The findings are descriptive for one base seed and this evaluation protocol; artifact and motor-response confounds remain.'])
    add('1. Research questions', [
        'The course questions [7] are: RQ1, which tested method gives the best single-trial frequent-versus-target classification? '
        'RQ2, does deep learning outperform standard machine learning under the chosen evaluation? '
        'RQ3, can the features used by the deep model be explained?',
        'We define the primary ranking by mean outer-fold ROC-AUC and also report balanced accuracy, target F1 and raw accuracy. '
        'Every prediction is made from one trial. Grand-average ERP waveforms visualize condition differences; the classifiers do not receive averages across trials.'])
    add('2. Dataset, scope and preprocessing', [
        'The dataset specified by the course is Nencki-Symfonia [1,2], retrieved from its public OpenNeuro copy ds004621 [3]. '
        'This experiment attempts all 42 oddball participants, sub-01 through sub-42, and retains all 42 after signal-quality criteria. '
        'S5 is standard/frequent (class 0), S6 is target (class 1). S7 distractors and response markers are not classification examples. '
        'REST, SRT and MSIT tasks are outside this binary classification question.',
        f'There are {before:,} standard/target events before rejection and {meta["n_trials"]:,} retained trials. '
        f'{dropped} trials are dropped ({100*dropped/before:.2f}%). No participant is excluded by the preset rule of requiring at least 10 retained trials per class. '
        'Retention of all subjects does not imply artifact-free EEG or perfect behavioral compliance. '
        'Sub-02, sub-14 and sub-17, flagged for behavioral task-compliance issues in the source paper, remain included; this is a limitation of the all-subject analysis.',
        f'Preprocessing uses MNE-Python {prep["mne"]}. Sixteen sensors are fixed in advance: '+', '.join(meta['channels'])+'. '
        'Continuous data are referenced to the average of these sensors and filtered from 0.5 to 30 Hz with a fourth-order Butterworth IIR using forward-backward filtering. '
        'Epochs run from -0.2 to 1.0 s with baseline correction over -0.2 to 0 s. '
        'An epoch is rejected if any selected channel has peak-to-peak amplitude above 150 microvolts or below 0.1 microvolts. '
        'Retained epochs are resampled to 128 Hz, giving 154 time points and 16 channels per trial.',
        'Interior acquisition boundaries are marked so filtering does not run across discontinuities and overlapping epochs are rejected. '
        'Affected recordings are '+', '.join(boundaries)+'. '
        'No ICA or manual component selection is performed. Residual ocular or muscle activity may remain; using only 16 sensors and fixed thresholds can miss artifacts. '
        'Forward-backward filtering uses future samples, so this is an offline analysis.'], 'erp.png')
    add('3. Features and models', [
        'Classical ML uses 20 successive 50 ms mean-amplitude windows from 0 to 1 s at each of 16 sensors, giving 320 features per trial. '
        'LDA and SVM use standardization fitted only on training trials. The majority baseline outputs training class priors.',
        'Shrinkage LDA uses automatic covariance shrinkage and equal class priors. RBF SVM uses balanced class weights, gamma=scale, '
        'and validation selects C from {0.1, 1, 10}; probability calibration is internal to training data. '
        'Random Forest uses 200 trees, minimum leaf size 3 and balanced class weights; validation selects maximum depth 6 or unlimited.',
        'The compact CNN is inspired by EEGNet [4] but is not an exact implementation. Eight temporal filters of length 33 precede '
        '16 grouped spatial filters spanning all sensors, batch normalization, ELU, average pooling and dropout 0.35. '
        'A depthwise temporal convolution of length 9 and a pointwise convolution precede adaptive pooling to eight time bins and a linear target logit. '
        'The CNN receives the entire -0.2 to 1.0 s epoch, whereas classical features are post-stimulus; this input difference is a limitation.',
        'Per-channel normalization is fitted on training trials and time samples only. CNN optimization uses AdamW with learning rate 0.001, '
        'weight decay 0.01, batch size 64 and gradient norm clipping at 5. '
        'Target-weighted binary cross-entropy uses the training standard/target ratio. '
        'Training allows 50 epochs and early-stopping patience 8 on validation AUC, then restores the best checkpoint. '
        'The selected epochs across five folds are '+', '.join(str(int(v)) for v in df[df.model=='Compact EEG CNN']['choice'])+'.'])
    membership = [['Fold','Train subjects','Validation','Test']] + [
        [str(s['fold']), str(len(s['train'])), str(len(s['validation'])), str(len(s['test']))] for s in splits]
    add('4. Subject-disjoint evaluation', [
        'Five GroupKFold outer splits hold out 8 or 9 people for testing. Seven other people form validation, and 26 or 27 form training. '
        'Validation membership is selected by a fixed rotation of sorted development subject IDs, approximately 20% of development subjects. '
        'Every retained subject and trial appears in an outer test set exactly once for each model. Exact memberships are in splits.json.',
        'Only validation AUC selects SVM C, forest depth and CNN checkpoint. Test labels are used after selection for evaluation and explanations. '
        'No test-fitted scaling, threshold tuning or oversampling is used. All methods share the same retained trials and splits. '
        'Classification uses a fixed threshold of 0.5. Class-weighted CNN sigmoid scores need not be calibrated probabilities.',
        'ROC-AUC measures discrimination across thresholds. Balanced accuracy is the mean of target sensitivity and standard specificity. '
        'Average precision summarizes the precision-recall curve; F1 is reported for the target class. '
        'We report mean ± sample standard deviation across outer folds. The folds share some training subjects; SD is descriptive, not a confidence interval. '
        'Ranking model families on their observed test means does not provide an independent post-selection performance guarantee.'],
        table=membership)
    add('5. Measured classification results', [
        f'{best} achieves the largest mean ROC-AUC, {avg.loc[best,"roc_auc"]:.3f} ± {sd.loc[best,"roc_auc"]:.3f}, '
        f'and balanced accuracy, {avg.loc[best,"balanced_accuracy"]:.3f} ± {sd.loc[best,"balanced_accuracy"]:.3f}. '
        f'The strongest classical method by mean AUC is {ml}, {avg.loc[ml,"roc_auc"]:.3f} ± {sd.loc[ml,"roc_auc"]:.3f}. '
        'CNN-minus-SVM AUC differences for the five paired outer folds are '+', '.join(f'{v:+.3f}' for v in differences)+'. '
        'All five favor the CNN in this run; this pattern alone is not a formal significance test or evidence of universal superiority.',
        f'Targets comprise {meta["n_target"]/meta["n_trials"]:.2%} of retained trials. '
        f'The majority classifier reaches {avg.loc["Majority","accuracy"]:.3f} accuracy while AUC and balanced accuracy remain 0.500. '
        f'SVM raw accuracy is {avg.loc["RBF SVM","accuracy"]:.3f}, above CNN accuracy {avg.loc["Compact EEG CNN","accuracy"]:.3f}, '
        'but its target recall and balanced accuracy are lower. Different metrics therefore answer different questions. '
        'Accuracy alone should not determine the result for this imbalanced task.',
        f'CNN target precision is {avg.loc["Compact EEG CNN","precision"]:.3f}, recall {avg.loc["Compact EEG CNN","recall"]:.3f}, '
        f'F1 {avg.loc["Compact EEG CNN","f1"]:.3f}, and average precision {avg.loc["Compact EEG CNN","average_precision"]:.3f}. '
        'The comparison plot shows fold means and SD; confusion matrices aggregate only out-of-fold predictions. '
        'ROC curves average foldwise curves to avoid treating differently calibrated fold scores as one common scale.'],
        'comparison.png', score_table)
    add('6. Explaining the CNN', [
        'Integrated gradients [5] approximate the target-logit gradient integral from the training channel-mean baseline to each held-out trial. '
        'We use 24 midpoint integration steps. Each fold uses equal numbers of standard and target held-out trials, at most 32 per class. '
        'Absolute attribution is averaged over these trials and then across folds. It shows magnitude, not the direction of evidence for target.',
        attribution,
        'The map shows prominent Pz sensitivity near 0.5 s and P3 sensitivity near 0.3 s. In the Pz ERP plot, targets are more positive than standards '
        'during the several-hundred-millisecond interval. These patterns are compatible with a P300-related condition difference, '
        'but reference choice affects voltage amplitude and polarity, and scalp attribution does not localize a neural generator.',
        'Temporal occlusion replaces one 100 ms post-stimulus window by the training channel mean and recalculates held-out AUC. '
        f'The greatest mean decrease is in {peak:.1f}-{peak+.1:.1f} s ({occ.loc[peak,"mean"]:.3f} ± {occ.loc[peak,"std"]:.3f}). '
        'The 0.2-0.3 s and 0.3-0.4 s windows also contribute. Abrupt masks can generate artificial inputs, so their effect is a sensitivity test.',
        'Targets require a button response, unlike standards. Predictions may therefore exploit motor-related activity, visual differences or residual artifacts '
        'alongside attention-related ERPs. These explanations are partial and baseline-dependent, not causal proof of P300-based decisions.'], 'attribution.png')
    add('7. Limitations and further work', [
        'This study includes the full 42-person oddball cohort, but remains limited to this dataset and one base seed. '
        'There are no repeated validation rotations, multiple CNN seeds, or independent external cohort. Hyperparameter search budgets differ by model family. '
        'The three participants with reported behavioral compliance issues remain included. Fixed amplitude rejection without ICA may leave artifacts.',
        'Useful extensions are repeated seeds, subject-level uncertainty estimates, sensitivity analysis excluding sub-02/sub-14/sub-17, '
        'matched post-stimulus-only CNN inputs, a pre-stimulus-only negative control, and pre-response windows to investigate motor confounding. '
        'These are proposed analyses, not completed experiments. Earlier eight-subject pilot scores are not a controlled ablation because cohort, validation sizes '
        'and training budget changed.'], 'occlusion.png')
    add('8. Answers to the assignment', [
        f'RQ1: The highest observed mean ROC-AUC among tested methods is {avg.loc[best,"roc_auc"]:.3f}, obtained by {best}, '
        f'with balanced accuracy {avg.loc[best,"balanced_accuracy"]:.3f}. This is not a global optimum over all possible methods.',
        f'RQ2: {conclusion} The result favors the CNN under the stated protocol, while SVM has higher raw accuracy at threshold 0.5. '
        'We do not claim deep learning is always better.',
        'RQ3: Integrated gradients and occlusion provide sensor/time explanations consistent with ERP differences. '
        'They do not isolate a causal physiological mechanism or exclude artifacts and motor-response information.'])
    add('9. Reproducibility and verification', [
        'The full experiment is implemented in kaggle/01_prepare_full.ipynb and 02_train_full.ipynb. '
        'The preparation notebook downloads one subject at a time and saves processed data, trial audits, original file hashes and acquisition-boundary information. '
        'The training notebook saves configuration, data checksum, split IDs, predictions, learning curves, CNN checkpoints and per-fold resume caches.',
        f'The recorded training environment is Python {meta["python"]}, NumPy {meta["numpy"]}, SciPy {meta["scipy"]}, '
        f'scikit-learn {meta["sklearn"]}, PyTorch {meta["torch"]}, with CNN CUDA execution on {meta["gpu_name"]}. '
        'The base seed is 42 and CNN fold seeds are base plus fold. Deterministic algorithms are requested with warn_only=True, so unsupported CUDA operations '
        'may remain nondeterministic. Hardware/software differences can change numerical results.',
        'The Kaggle verifier passes subject separation, complete out-of-fold prediction coverage, metric reconstruction, majority baseline checks '
        'and integrated-gradients numerical completeness. Locally, all seven reported metrics were independently recomputed from 121,000 predictions, '
        'and all five checkpoint file hashes matched their fold caches. The preparation summary agrees with training trial counts, subject IDs and input-data hash. '
        'The full epochs file is retained on Kaggle; it was not supplied locally for a second direct hash calculation.'])
    contributions = [['Member / ID','Proposed responsibility','Proposed %']] + [
        [f'{n} / {sid}', role, '20'] for n,sid,role in layout.MEMBERS]
    add('10. Group contributions and submission', [
        'The following equal allocation is a proposal, not a verified record of completed individual work. '
        'Each member must review their part, record actual work and demonstrated understanding, then confirm or revise the percentages before submission. '
        'The proposed total is 100%. Automated preparation and a name in the table do not establish individual contribution.',
        'The English presentation has ten main slides with a planned nine-minute script and two backup slides. '
        'All members should attend and prepare for questions. Send the English report at least one week before presentation to guy.nagels@vub.be, '
        'Mr Tran Quoc Sang and Mr Nguyen Tien Dung, using the latter two addresses supplied by the course. The presentation date has not been supplied.'],
        table=contributions)
    counts = [['Subject','Standard before','Target before','Standard kept','Target kept','Dropped']] + [
        [a['subject'],str(a['before']['standard']),str(a['before']['target']),str(a['after']['standard']),
         str(a['after']['target']),str(a['dropped'])] for a in prep['audits']]
    add('Appendix A. Trial audit for all participants', [
        f'Totals: {before:,} eligible standard/target events; {meta["n_trials"]:,} retained; {dropped} dropped. '
        'No participant failed the minimum per-class trial criterion. Behavioral compliance and signal quality are distinct.'], table=counts)
    add('References', refs)
    render_report(out, source, sections)
    layout.OUT, layout.RESULTS, layout.REFERENCES = out, source, refs
    slides = [list(x) for x in layout.build_slides(meta,avg,sd,best,ml,conclusion,delta,attribution,peak)]
    slides[0][1][0] = 'Nencki-Symfonia | all 42 oddball participants'
    slides[0][3] = slides[0][3].replace('This is an eight-subject pilot, so our conclusions are limited to this experiment.',
                                     'We analyze all forty-two participants in the oddball dataset, using subject-disjoint testing.')
    slides[2][3] = slides[2][3].replace('eight participants','forty-two participants')
    slides[2][1][0] = '24,200 retained trials; 240 rejected; all 42 participants retained'
    slides[4][1][:2] = ['5 outer folds: 8 or 9 test subjects', '26–27 train + 7 validation + 8–9 test subjects']
    slides[4][3] = ('We separate people in every outer fold. Five folds hold out eight or nine people each. '
                   'Seven other people form validation, leaving twenty-six or twenty-seven for training. '
                   'Validation selects the SVM penalty, forest depth and CNN checkpoint. The CNN uses temporal and grouped spatial convolutions inspired by EEGNet. '
                   'Every retained trial is tested exactly once per model. Scaling and class weights use training data only. '
                   'All methods share the same splits. Fold means and standard deviations describe performance and variability; they are not confidence intervals.')
    slides[6][1][2] = 'One base seed · 42 participants · limited search'
    slides[6][3] = slides[6][3].replace('use the remaining participants','use an independent cohort')
    slides[9][1][-1] = 'Next: repeated seeds, artifact and motor-response controls'
    slides[9][3] = slides[9][3].replace('under our pilot protocol','under our subject-disjoint protocol').replace(
        'Our next steps are the full cohort, repeated seeds and negative controls.',
        'Our next steps are repeated seeds, compliance sensitivity analysis and negative controls.')
    assert sum(s[4] for s in slides)==540
    layout.make_pptx(slides, project_label='Nencki-Symfonia EEG | 42 subjects')
    (out/'Presentation_script_EN.md').write_text('\n\n'.join(
        f'## Slide {i}: {s[0]} ({s[4]} s)\nSpeaker: {s[5]}\n\n{s[3]}' for i,s in enumerate(slides,1))+
        '\n\nTotal planned speaking time: 540 seconds (9 minutes).\n',encoding='utf-8')
    if not (out/'contributions.csv').exists():
        shutil.copy2('deliverables/contributions.csv',out/'contributions.csv')
    header = ('# Kết quả full cohort — thay cho các số liệu pilot trong hướng dẫn cũ\n\n'
              '42 người; 24.200 trial (20.801 standard, 3.399 target); loại 240 trial; không loại người theo QC.\n\n'
              'CNN: AUC 0,936 ± 0,031; balanced accuracy 0,866 ± 0,039; accuracy 0,887. '
              'SVM: AUC 0,907; accuracy 0,912. CNN tốt hơn theo AUC và balanced accuracy, SVM cao hơn theo accuracy.\n\n'
              '5 fold theo người: 26–27 train, 7 validation, 8–9 test. Một base seed, chưa kiểm định ý nghĩa thống kê. '
              'Pz/P3/O2 có tổng attribution lớn; che 0,4–0,5 s giảm AUC trung bình khoảng 0,0195. '
              'Không chứng minh tín hiệu chỉ là P300; còn nhiễu và thông tin vận động.\n\n'
              'Sử dụng Report_EN.pdf và Presentation_EN.pptx trong deliverables_full/. '
              'Giữ epochs_full.npz và preparation_summary.json từ notebook 01 để tái lập. '
              'Tỷ lệ đóng góp 20% mỗi người vẫn cần xác nhận theo công việc thực tế.\n')
    (out/'HUONG_DAN_FULL_VI.md').write_text(header,encoding='utf-8')
    print(f'Generated full-cohort report and presentation in {out}')

def render_report(out, source, sections):
    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(TTFont('FullSans', findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('FullSansBold',findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
    for name in ['Normal','BodyText','Title','Heading1','Heading2']:
        styles[name].fontName = 'FullSansBold' if name in ['Title','Heading1','Heading2'] else 'FullSans'
    styles['BodyText'].fontSize=10; styles['BodyText'].leading=14
    styles['Heading1'].keepWithNext=True
    cellstyle=ParagraphStyle('FullCell',parent=styles['BodyText'],fontSize=8.5,leading=11)
    doc=Document(); doc.styles['Normal'].font.name='Calibri'; doc.styles['Normal'].font.size=Pt(11)
    title='Single-Trial EEG Classification of Frequent and Target Stimuli'
    subtitle='Nencki-Symfonia | Full 42-participant oddball cohort'
    doc.add_heading(title,0); doc.add_paragraph(subtitle)
    pdf=[Paragraph(title,styles['Title']),Paragraph(subtitle,styles['Heading2']),Spacer(1,20)]
    markdown=[f'# {title}',subtitle]
    for name,sid,_ in layout.MEMBERS:
        doc.add_paragraph(f'{name} — {sid}')
        pdf.append(Paragraph(html.escape(f'{name} — {sid}'),styles['BodyText']))
        markdown.append(f'- {name} — {sid}')
    note='24,200 retained trials | Five subject-disjoint folds | CNN ROC-AUC 0.936 ± 0.031'
    doc.add_paragraph(note); pdf += [Spacer(1,20),Paragraph(note,styles['BodyText'])]
    doc.add_paragraph('Individual contribution allocation requires confirmation before submission.')
    pdf += [Spacer(1,15),Paragraph('Individual contribution allocation requires confirmation before submission.',styles['BodyText']),PageBreak()]
    doc.add_page_break()
    for title,paragraphs,figure,table in sections:
        if title.startswith('Appendix') or title=='References':
            doc.add_page_break(); pdf.append(PageBreak())
        doc.add_heading(title,1); pdf.append(Paragraph(html.escape(title),styles['Heading1']))
        markdown.append('## '+title)
        for text in paragraphs:
            doc.add_paragraph(text)
            pdf += [Paragraph(html.escape(text),styles['BodyText']),Spacer(1,7)]
            markdown.append(text)
        if table:
            layout.add_doc_table(doc,table)
            widths=([102,95,95,92,87] if len(table[0])==5 else
                    [70,80,80,80,80,81] if len(table[0])==6 else
                    [165,251,55] if len(table[0])==3 else [60,137,137,137])
            t=Table([[Paragraph(html.escape(str(x)),cellstyle) for x in row] for row in table],
                    colWidths=widths,repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8f3')),
                                  ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cbd5e1')),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            pdf += [t,Spacer(1,12)]
            markdown += ['| '+' | '.join(table[0])+' |','| '+' | '.join(['---']*len(table[0]))+' |']
            markdown += ['| '+' | '.join(row)+' |' for row in table[1:]]
        if figure:
            (out/'figures').mkdir(exist_ok=True)
            shutil.copy2(source/figure,out/'figures'/figure)
            doc.add_picture(str(source/figure),width=Inches(6.25))
            with PILImage.open(source/figure) as im: width,height=im.size
            pdf += [Image(str(source/figure),width=468,height=468*height/width),Spacer(1,12)]
            markdown.append(f'![{figure}](figures/{figure})')
    doc.save(out/'Report_EN.docx')
    (out/'Report_EN.md').write_text('\n\n'.join(markdown),encoding='utf-8')
    SimpleDocTemplate(str(out/'Report_EN.pdf'),leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48).build(
        pdf,onFirstPage=layout.page_number,onLaterPages=layout.page_number)

if __name__=='__main__':
    main()
