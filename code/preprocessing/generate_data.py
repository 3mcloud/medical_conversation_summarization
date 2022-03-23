#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:52:46 2020

@author: aah2azz

generate script for various ways of generating Oak dataset for summarization
"""
import argparse
import glob
import json
import jsonlines
import numpy as np
import os
import pandas as pd
import re

from pandarallel import pandarallel
from tqdm import tqdm


#%% utilities
def agg_stagex(grp, sep=' '):
    finalstr = []
    for sent in grp.values:
        if any([sent in x for x in finalstr]):
            continue
        finalstr.append(sent)
    return sep.join(finalstr)


def bart_preprocessing(x):
    return x.replace('\n', ' ').replace('\t', ' ')


def format_line(turn):
    return "[{}]: {}".format(turn["speaker_id"].upper(), re.sub(r'[\s]+', ' ', turn["utterance"]))


def get_conv(utters, no_role=False, rolemap=None):
    script = []
    for turn in utters:
        if rolemap is not None:
            sid = rolemap[turn['speaker_id']]
        else:
            sid = turn['speaker_id']
        utter = bart_preprocessing(turn['utterance']) 
        if no_role:
            script.append('{}'.format(utter))
        else:
            script.append('[{}]: {}'.format(sid, utter))
    script = ' '.join(script)
    return script


def get_mode(filename):
    if 'dev' in filename:
        mode = 'dev'
    elif 'test' in filename:
        mode = 'test'
    elif 'train' in filename:
        mode = 'train'
    else:
        raise ValueError("Can't infer dataset from file name, make sure train|dev|test is part of the filename")
    return mode


def get_sliding_snippets(utterances, win=8, stride=4):
    # create snippets by sliding window scan of converrsation with fixed # of turns
    turns = [format_line(x) for x in utterances]
    if len(turns) <= win:
        return [' '.join(turns),]
    snips = [' '.join(turns[i:i+win]) for i in range(0, len(turns)-win, stride)]
    return snips


def read_data(filename, cid='cid', stringify=True):
    """
    create a DataFrame from .jsonl|.json|.pckl file

    Inputs:
        filename (str) - file name
        stringify (bool, default = True) - whether to convert "physican" column to string
    """
    if filename.endswith('.jsonl'):
        df = []
        with jsonlines.open(filename, mode='r') as reader:
            for j in reader:
                df.append(j)
        df = pd.DataFrame(df)
    elif filename.endswith('.json'):
        df = pd.read_json(filename)
    elif filename.endswith('.pckl'):
        df = pd.read_pickle(filename)
    else:
        raise TypeError('Unrecognized file extension, supported are .jsonl|.json|.pckl')

    if stringify:
        df[cid] = df[cid].astype(str)
    df.sort_values(cid, inplace=True)
    return df


def save_file(df, folder, savefile, meta_cols=['cid', 'sid'], src_col=None, tgt_col=None):
    # save file utility
    if os.path.exists(folder):
        print(f"Warning! {folder}/ already exists, files with identical names will be overwritten")
    else:
        os.makedirs(folder)
    savefile = os.path.join(folder, savefile)
    df[meta_cols].to_csv(savefile+'.meta', sep='\t', index=True, header=True)
    print('saving meta file to {}'.format(savefile+'.meta'))
    if src_col is not None:
        with open(savefile+'.source', 'w') as writer:
            writer.write('\n'.join(df[src_col]))
        print('saving source file to {}'.format(savefile+'.source'))
    if tgt_col is not None:
        with open(savefile+'.target', 'w') as writer:
            writer.write('\n'.join(df[tgt_col]))
        print('saving target file to {}'.format(savefile+'.target'))


#%% helper functions for chunking method
def compute_break_index(utterance_lengths, start_idx, end_idx, overlap):
    assert overlap >= 0.0 and overlap <= 1.0
    num_utterances = end_idx - start_idx
    fragment_length = sum(utterance_lengths[start_idx:end_idx])
    running_length = 0
    idx = end_idx
    for i in range(num_utterances):
        running_overlap = running_length / float(fragment_length)
        if running_overlap >= overlap:
            return idx
        idx -= 1
        running_length += utterance_lengths[idx]
    return start_idx + 1


def chunk_conversation(x, header_length, fragment_length, fragment_overlap):
    num_utterances = len(x)
    utterance_lengths = []
    for i in range(len(x)):
        v = len(x[i]["utterance"].split())
        utterance_lengths.append(v)
    total_length = sum(utterance_lengths)
    header_utterances = []
    fragments = []
    if total_length <= header_length + fragment_length:
        header_utterances = x
    else:
        idx = 0
        h_length = 0
        # getting the header.
        while h_length + utterance_lengths[idx] <= header_length:
            header_utterances.append(x[idx])
            h_length += utterance_lengths[idx]
            idx += 1

        # TODO: check that I'm not off by one.
        # getting the fragments
        while idx < num_utterances:
            f_length = 0
            start_idx = idx
            fragment_utterances = []
            while idx < num_utterances and f_length + utterance_lengths[idx] <= fragment_length:
                fragment_utterances.append(x[idx])
                f_length += utterance_lengths[idx]
                idx += 1
            if len(fragment_utterances) >= 1:
                fragments.append(fragment_utterances)

            # prevent complete overlap.
            if idx < num_utterances:
                break_idx = compute_break_index(utterance_lengths, start_idx, idx, fragment_overlap)
                assert break_idx != start_idx
                idx = break_idx

    return header_utterances, fragments


def serialize_conversation_fragments(header_utterances, fragments, 
utterance_separator_str=' ', header_fragment_separator_str='...', continuation_str='...'):
    header_strs = []
    for u in header_utterances:
        s = format_line(u)
        header_strs.append(s)

    fragment_strs_lst = []
    for f in fragments:
        f_strs = []
        for u in f:
            s = format_line(u)
            f_strs.append(s)
        fragment_strs_lst.append(f_strs)

    out_strs = []
    header_str = utterance_separator_str.join(header_strs)
    if len(fragment_strs_lst) == 0:
        out_strs.append(header_str)
    else:
        num_fragments = len(fragment_strs_lst)
        assert num_fragments >= 1

        for i, f_strs in enumerate(fragment_strs_lst):
            fragment_str = utterance_separator_str.join(f_strs)

            # first fragment
            if i == 0:
                out_s = utterance_separator_str.join([header_str, fragment_str])
                if len(fragment_strs_lst) > 1:
                    out_s = utterance_separator_str.join([out_s, continuation_str])
            # inner fragments
            elif i < num_fragments - 1:
                out_s = utterance_separator_str.join([header_str, header_fragment_separator_str, fragment_str, continuation_str])
            # last fragment
            else:
                out_s = utterance_separator_str.join([header_str, header_fragment_separator_str, fragment_str])

            out_strs.append(out_s)
    return out_strs


#%% main APIs
def aggregate_between_stages(mode, expname, suffix='stagex', savefolder='../../experiments', sep=' ', grp_keys=['cid', 'sid']):
    savefolder = os.path.join(savefolder, expname)
    metafile = f'{mode}_{suffix}.meta'
    hypofile = f'{mode}_{suffix}.hypo'
    targetfile = f'{mode}_{suffix}.target'
    savefile = f'{mode}_stage2'

    dfmeta = pd.read_csv(os.path.join(savefolder, metafile), sep='\t', header=0, index_col=0)
    dfmeta['hypo'] = open(os.path.join(savefolder, hypofile)).read().strip().split('\n')
    dfmeta['target'] = open(os.path.join(savefolder, targetfile)).read().strip().split('\n')
    
    df = dfmeta.groupby(grp_keys).agg(
    {
        'hypo': lambda x: agg_stagex(x, sep=sep),
        'target': lambda x: x.to_list()[0],
    }).reset_index()
    
    save_file(df, savefolder, savefile, meta_cols=['cid', 'sid'], src_col='hypo', tgt_col='target')


def generate_chunk_data_stage1(filename, exp='', savefolder='../experiments/',
                               save=True, process_fn=None,
                               header_len=128, body_len=384, body_overlap=0.333,
                               **kwds):
    """
    generate data from chunking methods for multistage training - stage 1
    """
    df = read_data(filename)
    
    if process_fn is not None:
        df = process_fn(df, **kwds)
        
    snippets = []
    for (i, row) in df.iterrows():
        x = row['utterances']
        header_utterances, fragments = chunk_conversation(x, header_len, body_len, body_overlap)
        out_strs = serialize_conversation_fragments(header_utterances, fragments, utterance_separator_str=' ', header_fragment_separator_str='...', continuation_str='...')
        snippets.append(out_strs)
        # print(max([len(x.split()) for x in out_strs]))
    df['chunks'] = snippets
    dfout = df[['cid', 'sid', 'chunks', 'summary']].explode('chunks', ignore_index=True)
    dfout['summary'] = dfout['summary'].apply(bart_preprocessing)
    dfout['chunks'] = dfout['chunks'].apply(bart_preprocessing)

    if save:
        mode = get_mode(filename)
        folder = os.path.join(savefolder, exp)
        save_file(
            dfout, 
            folder,
            f'{mode}_stage1', 
            meta_cols=['cid', 'sid'], 
            src_col='chunks', 
            tgt_col='summary',
        )
        save_file(
            dfout, 
            folder,
            f'{mode}_stagex', 
            meta_cols=['cid', 'sid'], 
            src_col='chunks', 
            tgt_col='summary',
        )

    return dfout


def generate_data_vanilla(filename, exp='', meta_cols=['cid', 'sid'], save=True, savefolder='../experiments/', process_fn=None, **kwds):  
    df = read_data(filename)
    if process_fn is not None:
        df = process_fn(df, **kwds)
    else:
        if 'conv' not in df.columns:
            df['conv'] = df['utterances'].parallel_apply(get_conv)
        else:
            df['conv'] = df['conv'].parallel_apply(bart_preprocessing)
        df['summary'] = df['summary'].parallel_apply(bart_preprocessing)
    
    if save:
        mode = get_mode(filename)
        folder = os.path.join(savefolder, exp)
        save_file(
            df, 
            folder,
            f'{mode}',
            meta_cols=meta_cols, 
            src_col='conv', 
            tgt_col='summary',
        )

    return df


def generate_data_multistage_stage1(file, exp, savefolder='../experiments', save=True, win=8, stride=4, meta_cols=['cid', 'sid'], process_fn=None, **kwds):
    df = read_data(file)
    if process_fn is not None:
        df = process_fn(df, **kwds)
    else:
        df['snippet'] = df['utterances'].parallel_apply(get_sliding_snippets, win=win, stride=stride)
        df['summary'] = df['summary'].parallel_apply(bart_preprocessing)

        dfout = df[meta_cols + ['snippet', 'summary']].explode('snippet', ignore_index=True)
    
    if save:
        mode = get_mode(file)
        folder = os.path.join(savefolder, exp)
        save_file(
            dfout, 
            folder,
            f'{mode}_stagex',
            meta_cols=['cid', 'sid'], 
            src_col='snippet', 
            tgt_col='summary',
        )

    return df


#%% input parser
def cli_parse():
    parser = argparse.ArgumentParser(description='Data generation for stage 1 training of multistage chunking method')
    
    parser.add_argument('--file', type=str, default=None,
                        help='path to .jsonl file of input data')
    parser.add_argument('--exp', type=str, required=True,
                        help='name of experiment, create subfolder under experiments/ folder')
    parser.add_argument('--mode', type=str, required=True, choices=['chunk', 'plain', 'multistage', 'aggregate'],
                        help='type of data generation, "chunk" or "plain" or "multistage"')
    parser.add_argument('--no_save', action='store_true',
                        help="do not save files")
    parser.add_argument('--meta_cols', nargs='+', default=['cid', 'sid'],
                        help="List of meta columns in data files")
    # for chunking methods
    parser.add_argument('--header_len', type=int, default=128,
                        help="# of words in the header of each chunk - for multistage chunking method")
    parser.add_argument('--body_len', type=int, default=128,
                        help="# of words in the body of each chunk - for multistage chunking method")
    parser.add_argument('--body_overlap', type=float, default=0.3333,
                        help="# of words overlapped in the body of adjacent chunks - for multistage chunking method")
    # for sentbert methods
    parser.add_argument('--snippet_window', type=int, default=8,
                        help="# of utterance in each snippet - for multistage sentbert method")
    parser.add_argument('--snippet_stride', type=int, default=6,
                        help="# of utterance in each snippet - for multistage sentbert method")                  
    # for data aggregation between stages
    parser.add_argument('--agg_suffix', type=str,  default='stagex',
                        help='name suffix of data files to be aggregated')
    parser.add_argument('--agg_keys', nargs='+', default=['cid', 'sid'],
                        help="List of id keys used in grouping similar inputs, could be different from --meta_cols")
    parser.add_argument('--agg_sep', type=str, default=' ',
                        help="Separator used in aggregating stage 1 summaries")
    parser.add_argument('--agg_dataset', nargs='+', default=['dev', 'test', 'train'],
                        help="Types of datasets to process, default is ['dev', 'test', 'train']")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get the correct folder for saving data files
    savefolder = 'experiments/'
    while not os.path.isdir(savefolder):
        savefolder = os.path.join('..', savefolder)

    pandarallel.initialize()

    args = cli_parse()

    if args.mode != 'aggregate' and args.file is None:
        raise ValueError('--file argument must be specified for data generation')

    if args.mode == 'chunk':
        generate_chunk_data_stage1(
            args.file, 
            exp=args.exp,
            meta_cols=args.meta_cols,
            save=not args.no_save, 
            savefolder=savefolder,
            header_len=args.header_len, 
            body_len=args.body_len, 
            body_overlap=args.body_overlap,
        )
    elif args.mode == 'plain':
        generate_data_vanilla(
            args.file, 
            exp=args.exp, 
            meta_cols=args.meta_cols,
            savefolder=savefolder,
            save=not args.no_save, 
            inbetween_stage=False,
        )
    elif args.mode == 'multistage':
        generate_data_multistage_stage1(
            args.file, 
            exp=args.exp, 
            meta_cols=args.meta_cols,
            savefolder=savefolder,
            save=not args.no_save, 
            win=args.snippet_window,
            stride=args.snippet_stride,
        )
    elif args.mode == 'aggregate':
        for mode in args.agg_dataset:
            aggregate_between_stages(
                mode, 
                args.exp, 
                suffix=args.agg_suffix, 
                savefolder=savefolder,
                sep=args.agg_sep, 
                grp_keys=args.agg_keys,
            )