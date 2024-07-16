import pandas as pd
import plotly
import os

if __name__ == '__main__':
    bs_context_path = 'results/bertscore_context.tsv'
    bs_simple_path = 'results/bertscore_simple.tsv'
    bs_transl_path = 'results/bertscore_transl.tsv'

    bl_context_path = 'results/bleu_context.tsv'
    bl_simple_path = 'results/bleu_simple.tsv'
    bl_transl_path = 'results/bleu_transl.tsv'

    c_context_path = 'results/cider_context.tsv'
    c_simple_path = 'results/cider_simple.tsv'
    c_transl_path = 'results/cider_transl.tsv'

    m_context_path = 'results/meteor_context.tsv'
    m_simple_path = 'results/meteor_simple.tsv'
    m_transl_path = 'results/meteor_transl.tsv'

    r_context_path = 'results/rouge_context.tsv'
    r_simple_path = 'results/rouge_simple.tsv'
    r_transl_path = 'results/rouge_transl.tsv'

    paths = [bs_context_path, bs_simple_path, bs_transl_path, bl_context_path, bl_simple_path, bl_transl_path,
             c_context_path, c_simple_path, c_transl_path, m_context_path, m_simple_path, m_transl_path,
             r_context_path, r_simple_path, r_transl_path]

    dfs = []
    for path in paths:
        style = path.split('_')[1].split('.')[0]
        df = pd.read_csv(path, sep='\t')
        df.assign(Prompt= [style for i in range(len(df.index))])
        dfs.append(df)
    print(dfs[0])
