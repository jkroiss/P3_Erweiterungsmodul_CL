import pandas as pd
from collections import defaultdict
import plotly.express as px
import os

if __name__ == '__main__':
    bs_context_path = 'bertscore_context.tsv'
    bs_simple_path = 'bertscore_simple.tsv'
    bs_transl_path = 'bertscore_transl.tsv'

    #bl_context_path = 'bleu_context.tsv'
    #bl_simple_path = 'bleu_simple.tsv'
    #bl_transl_path = 'bleu_transl.tsv'

    #c_context_path = 'cider_context.tsv'
    #c_simple_path = 'cider_simple.tsv'
    #c_transl_path = 'cider_transl.tsv'

    m_context_path = 'meteor_context.tsv'
    m_simple_path = 'meteor_simple.tsv'
    m_transl_path = 'meteor_transl.tsv'

    r_context_path = 'rouge_context.tsv'
    r_simple_path = 'rouge_simple.tsv'
    r_transl_path = 'rouge_transl.tsv'

    paths = [bs_context_path, bs_simple_path, bs_transl_path, m_context_path, m_simple_path, m_transl_path,
             r_context_path, r_simple_path, r_transl_path]

    dfs = defaultdict(list)
    for path in paths:
        style = path.split('_')[1].split('.')[0]
        metric = path.split('_')[0]
        df = pd.read_csv(path, sep='\t')
        df = df.assign(Prompt= [style for i in range(len(df.index))])
        dfs[metric].append(df)

    concat_frames = {}
    for metric in dfs:
        frame = pd.concat(dfs[metric])
        concat_frames[metric] = frame

    #concat_frames['bleu'].drop(columns=['precisions'], inplace=True)
    #concat_frames['bleu'].drop(index=0, inplace=True)
    concat_frames['bertscore'].drop(columns=['prec', 'rec'], inplace=True)
    concat_frames['bertscore'].drop(index=0, inplace=True)
    concat_frames['rouge'].drop(columns=['rouge1', 'rouge2', 'rougeLsum'], inplace=True)
    concat_frames['rouge'].drop(index=0, inplace=True)

    fig = px.scatter(concat_frames['bertscore'], x='lang', y='f1', color='Prompt').update_traces(mode='lines+markers')
    fig.update_traces(marker_size=15)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.show()

    #fig = px.scatter(concat_frames['bleu'], x='lang', y='bleu', color='Prompt').update_traces(mode='lines+markers')
    #fig.update_traces(marker_size=15)
    #fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    #fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    #fig.show()

    #fig = px.scatter(concat_frames['cider'], x='lang', y='avg_score', color='Prompt').update_traces(mode='lines+markers')
    #fig.update_traces(marker_size=15)
    #fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    #fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    #fig.show()

    fig = px.scatter(concat_frames['meteor'], x='lang', y='meteor', color='Prompt').update_traces(mode='lines+markers')
    fig.update_traces(marker_size=15)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.show()

    fig = px.scatter(concat_frames['rouge'], x='lang', y='rougeL', color='Prompt').update_traces(mode='lines+markers')
    fig.update_traces(marker_size=15)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.show()


