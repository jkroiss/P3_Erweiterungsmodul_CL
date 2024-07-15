#!/usr/bin/python3
import os
from IPython.display import display
import evaluate
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    gens, refs, gen_ref_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    path = 'results/mBlip_simple_prompt.tsv'
    df = pd.read_csv(path, sep='\t')
    for lang in range(len(df.index)):
        gens[df.iloc[lang]['lang']].append(df.iloc[lang]['gen_caption'])
        refs[df.iloc[lang]['lang']].append(df.iloc[lang]['ref_caption'])
    for key in gens:
        gen_ref_dict[key].append(gens[key])
        gen_ref_dict[key].append(refs[key])

    #Apply and save bertscore for each langauge
    bertscore = evaluate.load('bertscore')
    df_bertscore = pd.DataFrame(columns=['lang','prec', 'rec', 'f1'])
    i = 0
    for lang in gen_ref_dict:
        pred = gen_ref_dict[lang][0]
        ref = gen_ref_dict[lang][1]
        output = bertscore.compute(predictions=pred, references = ref, lang=lang)
        p, r, f1 = output['precision'], output['recall'], output['f1']
        df_bertscore.loc[i] = [lang, sum(p)/len(p), sum(r)/len(r), sum(f1)/len(f1)]
        i += 1

    df_bertscore.to_csv('results/bertscore.tsv', sep='\t')



    #rouge = evaluate.load('rouge')
    #bleu = evaluate.load('bleu')
    #meteor = evaluate.load('meteor')
