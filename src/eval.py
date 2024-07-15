#!/usr/bin/python3
import os
from IPython.display import display
import evaluate
import pandas as pd
from collections import defaultdict
#from cidereval import cider
from pycocoevalcap.cider.cider import Cider
from bleurt import score


if __name__ == '__main__':

    gens, refs, gen_ref_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    path_to_simple = 'results/mBlip_simple_prompt.tsv'
    path_to_context = 'results/mBlip_context_prompt.tsv'
    path_to_transl = 'results/mBlip_context_prompt.tsv'

    df_simple = pd.read_csv(path_to_simple, sep='\t')
    df_context = pd.read_csv(path_to_context, sep='\t')
    df_transl = pd.read_csv(path_to_transl, sep='\t')

    for lang in range(len(df_simple.index)):
        gens[df_simple.iloc[lang]['lang']].append(df_simple.iloc[lang]['gen_caption'])
        refs[df_simple.iloc[lang]['lang']].append(df_simple.iloc[lang]['ref_caption'])
    for key in gens:
        gen_ref_dict[key].append(gens[key])
        gen_ref_dict[key].append(refs[key])

    # Set up dataframes
    df_rouge = pd.DataFrame(columns=['lang', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    df_bleu = pd.DataFrame(columns=['lang', 'bleu', 'precisions'])
    df_meteor = pd.DataFrame(columns=['lang', 'meteor'])
    #df_cider = pd.DataFrame(columns=['lang', 'avg_score', 'scores'])
    df_bertscore = pd.DataFrame(columns=['lang','prec', 'rec', 'f1'])

    # Load metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load('bertscore')

    i = 0
    for lang in gen_ref_dict:
        pred = gen_ref_dict[lang][0]
        ref = gen_ref_dict[lang][1]

        r_output = rouge.compute(predictions=pred, references=ref)
        df_rouge.loc[i] = [lang, r_output['rouge1'], r_output['rouge2'], r_output['rougeL'], r_output['rougeLsum']]

        b_output = bleu.compute(predictions=pred, references=ref)
        df_bleu.loc[i] = [lang, b_output['bleu'], b_output['precisions']]

        m_output = meteor.compute(predictions=pred, references=ref)
        df_meteor.loc[i] = [lang, m_output['meteor']]

        #c_output = cider(predictions=pred, references=ref)
        #df_cider.loc[i] = [lang, c_output['avg_score'], c_output['scores']]

        bs_output = bertscore.compute(predictions=pred, references = ref, lang=lang)
        p, r, f1 = bs_output['precision'], bs_output['recall'], bs_output['f1']
        df_bertscore.loc[i] = [lang, sum(p)/len(p), sum(r)/len(r), sum(f1)/len(f1)]


        i += 1

    df_rouge.to_csv('results/rouge_simple.tsv', sep='\t')
    df_bleu.to_csv('results/bleu_simple.tsv', sep='\t')
    df_meteor.to_csv('results/meteor_simple.tsv', sep='\t')
    #df_cider.to_csv('results/cider_simple.tsv', sep='\t')
    df_bertscore.to_csv('results/bertscore_simple.tsv', sep='\t')

    gens, refs, gen_ref_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    for lang in range(len(df_context.index)):
        gens[df_context.iloc[lang]['lang']].append(df_context.iloc[lang]['gen_caption'])
        refs[df_context.iloc[lang]['lang']].append(df_context.iloc[lang]['ref_caption'])
    for key in gens:
        gen_ref_dict[key].append(gens[key])
        gen_ref_dict[key].append(refs[key])

   # Set up dataframes
    df_rouge = pd.DataFrame(columns=['lang', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    df_bleu = pd.DataFrame(columns=['lang', 'bleu', 'precisions'])
    df_meteor = pd.DataFrame(columns=['lang', 'meteor'])
    #df_cider = pd.DataFrame(columns=['lang', 'avg_score', 'scores'])
    df_bertscore = pd.DataFrame(columns=['lang', 'prec', 'rec', 'f1'])

    i = 0
    for lang in gen_ref_dict:
        pred = gen_ref_dict[lang][0]
        ref = gen_ref_dict[lang][1]

        r_output = rouge.compute(predictions=pred, references=ref)
        df_rouge.loc[i] = [lang, r_output['rouge1'], r_output['rouge2'], r_output['rougeL'], r_output['rougeLsum']]

        b_output = bleu.compute(predictions=pred, references=ref)
        df_bleu.loc[i] = [lang, b_output['bleu'], b_output['precisions']]

        m_output = meteor.compute(predictions=pred, references=ref)
        df_meteor.loc[i] = [lang, m_output['meteor']]

        #c_output = cider(predictions=pred, references=ref)
        #df_cider.loc[i] = [lang, c_output['avg_score'], c_output['scores']]

        bs_output = bertscore.compute(predictions=pred, references=ref, lang=lang)
        p, r, f1 = bs_output['precision'], bs_output['recall'], bs_output['f1']
        df_bertscore.loc[i] = [lang, sum(p) / len(p), sum(r) / len(r), sum(f1) / len(f1)]

        i += 1

    df_rouge.to_csv('results/rouge_context.tsv', sep='\t')
    df_bleu.to_csv('results/bleu_context.tsv', sep='\t')
    df_meteor.to_csv('results/meteor_context.tsv', sep='\t')
    #df_cider.to_csv('results/cider_context.tsv', sep='\t')
    df_bertscore.to_csv('results/bertscore_context.tsv', sep='\t')

    gens, refs, gen_ref_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for lang in range(len(df_transl.index)):
        gens[df_transl.iloc[lang]['lang']].append(df_transl.iloc[lang]['gen_caption'])
        refs[df_transl.iloc[lang]['lang']].append(df_transl.iloc[lang]['ref_caption'])
    for key in gens:
        gen_ref_dict[key].append(gens[key])
        gen_ref_dict[key].append(refs[key])


    df_rouge = pd.DataFrame(columns=['lang', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    df_bleu = pd.DataFrame(columns=['lang', 'bleu', 'precisions'])
    df_meteor = pd.DataFrame(columns=['lang', 'meteor'])
    #df_cider = pd.DataFrame(columns=['lang', 'avg_score', 'scores'])
    df_bertscore = pd.DataFrame(columns=['lang', 'prec', 'rec', 'f1'])

    i = 0
    for lang in gen_ref_dict:
        pred = gen_ref_dict[lang][0]
        ref = gen_ref_dict[lang][1]

        r_output = rouge.compute(predictions=pred, references=ref)
        df_rouge.loc[i] = [lang, r_output['rouge1'], r_output['rouge2'], r_output['rougeL'], r_output['rougeLsum']]

        b_output = bleu.compute(predictions=pred, references=ref)
        df_bleu.loc[i] = [lang, b_output['bleu'], b_output['precisions']]

        m_output = meteor.compute(predictions=pred, references=ref)
        df_meteor.loc[i] = [lang, m_output['meteor']]

        #c_output = cider(predictions=pred, references=ref)
        #df_cider.loc[i] = [lang, c_output['avg_score'], c_output['scores']]

        bs_output = bertscore.compute(predictions=pred, references=ref, lang=lang)
        p, r, f1 = bs_output['precision'], bs_output['recall'], bs_output['f1']
        df_bertscore.loc[i] = [lang, sum(p) / len(p), sum(r) / len(r), sum(f1) / len(f1)]

        i += 1

    df_rouge.to_csv('results/rouge_transl.tsv', sep='\t')
    df_bleu.to_csv('results/bleu_transl.tsv', sep='\t')
    df_meteor.to_csv('results/meteor_transl.tsv', sep='\t')
    #df_cider.to_csv('results/cider_transl.tsv', sep='\t')
    df_bertscore.to_csv('results/bertscore_transl.tsv', sep='\t')



