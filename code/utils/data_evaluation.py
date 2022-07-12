# -*- coding: utf-8 -*-
import jsonlines
import numpy as np
import os
import pandas as pd
pd.set_option("display.precision", 4)
import rouge
SCORER = rouge.Rouge()
from rouge_score import rouge_scorer
RSCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

from collections import defaultdict, Counter
from pandarallel import pandarallel
pandarallel.initialize()
from quickumls import QuickUMLS  # uncomment if quickUMLS is installed
from tqdm import tqdm


#%% Helper Functions
def config_UMLSmatcher(path):
    """Create UMLS matcher."""
    overlapping_criteria = 'length'
    threshold = 0.8
    similarity_name = 'jaccard'
    window = 5
    accepted_semtypes = None
    matcher = QuickUMLS(path, 
                        overlapping_criteria=overlapping_criteria, 
                        threshold=threshold,
                        similarity_name=similarity_name,
                        window=window)
    return matcher


def create_dataframe_from_jsonline(filename, cid='cid'):
    """Create pandas.DataFrame object from .jsonl files."""
    dfval = []
    with jsonlines.open(filename, mode='r') as reader:
        for j in reader:
            dfval.append(j)
    dfval = pd.DataFrame(dfval)
    dfval[cid] = dfval[cid].astype(str)
    return dfval


def get_scores(hyps, refs, avg=False):
    """Calculate rouge scores given a list of hypotheses and references. Modify this function but keep the function signature, if using different ROUGE package."""
    scores = defaultdict(list)
    for (h, r) in zip(hyps, refs):
        s = RSCORER.score(r, h)
        for key in s:
            scores[key+'_p'].append(s[key].precision)
            scores[key+'_r'].append(s[key].recall)
            scores[key+'_f'].append(s[key].fmeasure)
    
    if avg:
        for key in scores:
            scores[key] = np.mean(scores[key])
    
    return scores
    

def calculate_rouge(df, **kwds):
    """Wrapper function for calculating rouge scores on pandas.DataFrame object."""
    df['scores'] = df[['summary', 'hypo']].parallel_apply(apply_rouge_scores, axis=1, **kwds)
    scores = pd.DataFrame(df['scores'].tolist()).mean()
    return df, scores


def calculate_rouge_refs(df, **kwds):
    """Wrapper function for calculation rouge scores among references."""
    def apply_fn(refs):
        scores = []
        for i in range(len(refs)):
            h = [refs[i],] * (len(refs)-1)
            r = refs[:i]+refs[i+1:]
            try:
                scores.append(get_scores(h, r, avg=True))
            except:
                continue
        return pd.DataFrame(scores).mean()
    
    output = df['summary'].parallel_apply(apply_fn)
    return output


def apply_rouge_scores(value, return_type='mean', on='rouge1_f'):
    """'apply' type of function to be used in pandas.DataFrame.apply() method for calculating rouge scores."""
    assert(return_type in ['best', 'mean', 'worst'])
    refs = value['summary']
    hypos = [value['hypo']]*len(refs)
    scores = get_scores(hypos, refs, avg=False)
    d = pd.DataFrame(scores)
    if return_type == 'best':
        return d.iloc[d[on].argmax()].to_dict()
    elif return_type == 'worst':
        return d.iloc[d[on].argmin()].to_dict()
    else:
        return d.mean().to_dict()


def apply_extract_umls_concepts(value, umlsmatcher, return_type='dict'):
    """'apply' type of function to be used in pandas.DataFrame.apply() method for calculating quickUMLS concept metrics."""
    concepts = umlsmatcher.match(value)
    cuis = defaultdict(list)
    for item in concepts:
        for con in item:
            cuis[con['cui']].append(con['term'])
    if return_type == 'set':
        return set(cuis.keys())
    else:
        return cuis
    

def agg_merge_umls_concept(value, majority=None):
    """'agg' type of function to be used in pandas.DataFrame.agg() method for calculating quickUMLS concept metrics."""
    cui_cnts = Counter()
    for x in value:
        cui_cnts.update(x.keys())
    
    if majority is None:
        mv = 0
    elif majority == 'mv':
        mv = len(value) // 2
    else:
        mv = min(majority, len(value))
        
    out = set([key for key in cui_cnts if cui_cnts[key] >= mv])
    
    return out


def apply_naive_evaluation(value, refkey='umls', hypokey='umls_hypo', eps=1e-6):
    """Helper function used in quickUMLS concept-based evaluation."""
    refs = value[refkey]
    hypos = value[hypokey]
    intersects = refs.intersection(hypos)
    P = len(intersects)/(len(hypos) + eps)
    R = len(intersects)/(len(refs) + eps)
    F1 = 2*P*R/(P+R+eps)
    return F1, P, R


def normalize_input_files(hypo_files, meta_files, concept_folders=None):
    """Normalize the format of hypothesis (generated summaries) files for evaluation.
    
    Required parameters
    -------------------
    hypo_files: str or list[str], a single hypothesis (generated summaries) file or a list of hypothesis files.
    meta_files: str or list[str], a single .meta file or a list of .meta files associated with <hypo_files>.

    Keyword parameters
    ------------------
    concept_folders: str or list[str] (default None), optional parameter specifying path to folders containing concepts extracted from summaries, based on in-house NLP engine.

    Returns
    -------
    A zipped object containing paired (hypo, meta) file paths or paired (hypo, meta, concept_folder) file paths.
    """
    if type(hypo_files) is str:
        hypo_files = [hypo_files, ]
    if type(meta_files) is str:
        meta_files = [meta_files,] * len(hypo_files)
    if len(meta_files) != len(hypo_files):
        raise ValueError("meta file list and hypo file list don't contain the same number of files")
    for file in meta_files:
        if not os.path.exists(file):
            raise FileExistsError(f'{file} does not exit')
    for file in hypo_files:
        if not os.path.exists(file):
            raise FileExistsError(f'{file} does not exit')
    if concept_folders is not None:
        if type(concept_folders) is str:
            concept_folders = [concept_folders,] * len(hypo_files)
        if len(concept_folders) != len(hypo_files):
            raise ValueError("# concept folders and # hypo file list don't match")
        return zip(hypo_files, meta_files, concept_folders)
    else:
        return zip(hypo_files, meta_files)


def normalize_ref_files(ref_file, cid='cid', uid='utterances'):
    """Normalizing the format of reference data files for evaluation."""
    if ref_file.endswith('.jsonl'):
        dfval = create_dataframe_from_jsonline(ref_file)
    elif ref_file.endswith('.target'):
        ref_meta_file = ref_file.replace('.target', '.meta')
        if not os.path.exists(ref_meta_file):
            raise FileNotFoundError('{} does not exist'.format(ref_meta_file))
        dfval = pd.read_csv(ref_meta_file, sep='\t', header=0, index_col=0)
        dfval['summary'] = open(ref_file).read().strip().split('\n')
    else:
        raise TypeError('Unrecognized file extension for reference input, must be .jsonl or .target')
    
    agg_funs = {}
    for col in dfval.columns:
        if col == uid:
            agg_funs[col] = lambda x: x.iloc[0]
        elif col != cid:
            agg_funs[col] = lambda x: x.tolist()
            
    dfval = dfval.groupby(cid).agg(agg_funs).reset_index()
    
    return dfval


def read_meta_file(meta_file, hypo_file, cid='cid'):
    """Load data from individual hypothesis file as pandas.DataFrame object."""
    if meta_file.endswith('.jsonl'):
        df = create_dataframe_from_jsonline(meta_file)
    elif meta_file.endswith('.meta'):
        df = pd.read_csv(meta_file, sep='\t', index_col=0, header=0)
        df[cid] = df[cid].astype(str)
    else:
        raise TypeError("Invalid extention for meta file, must be .jsonl or .meta")
    df['hypo'] = open(hypo_file).read().strip().split('\n')
    df = df[[cid, 'hypo']].drop_duplicates()
    return df


#%% Main APIs
def eval_ref(jsonl_file,  cid='cid', process_fn=None, **kwds):
    """Evaluate mean-of-mean and mean-of-best rouge scores among reference summaries.

    Required parameters
    -------------------
    jsonl_file: str, path to reference .jsonl file.

    Keyword parameters
    ------------------
    cid : str (default 'cid'), name of the column for conversation identifier.
    process_fn : function (default None), additional preprocessing function that takes in and returns a pandas.DataFrame object.
    **kwds : dict, additional keyword parameters supported by process_fn().

    Returns
    -------
    (mean-of-mean, mean-of-best) rouge scores.
    """
    dfref = create_dataframe_from_jsonline(jsonl_file, cid=cid)
    dfscore = dfref[[cid, 'summary']].groupby(cid).agg(lambda x: x.to_list()).reset_index()
    if process_fn is not None:
        dfscore['summary'] = dfscore['summary'].parallel_apply(process_fn)
    mom = []
    mob = []
    for i in tqdm(range(dfscore.shape[0])):
        summs = dfscore.iloc[i]['summary']
        if len(summs) > 1:
            scores = []
            scores_best = []
            for i in range(len(summs)):
                h = [summs[i],] * (len(summs)-1)
                r = summs[:i]+summs[i+1:]

                s = pd.DataFrame(get_scores(h, r, avg=False))
                idx = np.argsort(s['rouge1_f'].values)  # sort by rouge-1 F1 scores

                scores.append(s.mean())
                scores_best.append(s.iloc[idx[-1]])
            mob.append(sum(scores_best)/len(scores_best))
            mom.append(sum(scores)/len(scores))
    return sum(mom)/len(mom), sum(mob)/len(mob)

    
def eval_rouge(hypo_files, meta_files, ref_file, cid='cid', process_fn=None, remove_duplicate=False):
    """Evaluate mean-of-mean and mean-of-best rouge scores for generated summaries.

    Required parameters
    -------------------
    hypo_files: str or list[str], a single .hypo file or a list of .hypo files, each containing generated summaries.
    meta_files: str or list[str], a single .meta file or a list of .meta files, each containing meta file for the corresponding .hypo file in <hypo_file>.
    ref_file: str, path to .jsonl file containing reference summaries.

    Keyword parameters
    ------------------
    cid : str (default 'cid'), name of the column for conversation identifier.
    remove_duplicate: bool (default False), whether to remove duplicated generated summaries from evaluation. Set this to True if the .hypo file contains multiple copies of generated summaries for the same conversation.
    process_fn : function (default None), additional preprocessing function that takes in and returns a pandas.DataFrame object.
    **kwds : dict, additional keyword parameters supported by process_fn().

    Returns
    -------
    A three-tuple: (dfout, mean-of-mean, mean-of-best), containing the processed pandas.DataFrame object (text and rouge scores) of the last .hypo file in the input, a list of mean-of-mean rouge scores,
    and a list of mean-of-best rouge scores. The size of the list matches the number of input .hypo files.
    """
    files = normalize_input_files(hypo_files, meta_files)
        
    dfval = normalize_ref_files(ref_file)
    
    scores_best = []
    scores_mean = []
    for (hypo_file, meta_file) in files:
        dfhypo = read_meta_file(meta_file, hypo_file, cid=cid)
        merge_cols = [cid]
        if remove_duplicate:
            dfhypo = dfhypo.groupby(merge_cols).agg(lambda x: x.iloc[0]).reset_index()
        df = dfval.merge(dfhypo, on=merge_cols, how='inner')
        assert(df.shape[0] == dfhypo.shape[0])  
        assert(df.shape[0] == df['summary'].notnull().sum())
        assert(df.shape[0] == df['hypo'].notnull().sum())
        
        if process_fn is not None:
            df['summary'] = df['summary'].parallel_apply(process_fn)
            df['hypo'] = df['hypo'].parallel_apply(process_fn)
        
        _, s_best = calculate_rouge(df, return_type='best', on='rouge1_f')
        scores_best.append(s_best)
        dfout, s_mean = calculate_rouge(df, return_type='mean')
        scores_mean.append(s_mean)
        
    if len(scores_mean) == 1:
        return dfout, s_mean, s_best
    else:
        return dfout, scores_mean, scores_best


def eval_umls(hypo_files, meta_files, ref_file, matcher, majority=3, cid='cid', sid='sid', remove_duplicate=False):
    """Evaluate quickUMLS concept-based evaluation for generated summaries.

    Required parameters
    -------------------
    hypo_files: str or list[str], a single .hypo file or a list of .hypo files, each containing generated summaries.
    meta_files: str or list[str], a single .meta file or a list of .meta files, each containing meta file for the corresponding .hypo file in <hypo_file>.
    ref_file: str, path to .jsonl file containing reference summaries.
    matcher: object, quickUMLS matcher object, can be set by config_UMLSmatcher().

    Keyword parameters
    ------------------
    cid: str (default 'cid'), name of the column for conversation identifier (unique for each conversation).
    sid: str (default 'sid'), name of the column for summary identifier (unique for each reference summary).
    remove_duplicate: bool (default False), whether to remove duplicated generated summaries from evaluation. Set this to True if the .hypo file contains multiple copies of generated summaries for the same conversation.
    majority: int (default 3), threshold for majority voting of extracted concepts from reference summaries. Only concepts that appear more than min(majority, # of references) times among all references are kept as gold concepts.

    Returns
    -------
    A tuple: (df, metrics), containing the processed pandas.DataFrame object (text and umls scores) of the last .hypo file in the input, and a list of UMLS metrics evaluated on individual conversations; the size of the list is equal
    to the number of input .hypo files. Use np.mean(metrics[i], axis=1) to get the mean scores for the ith .hypo file.
    """
    files = normalize_input_files(hypo_files, meta_files)
        
    # get reference concepts
    dfref = create_dataframe_from_jsonline(ref_file, cid=cid)
    concepts = []
    for i in tqdm(range(dfref.shape[0])):
        s = dfref['summary'].iloc[i]
        concepts.append(apply_extract_umls_concepts(s, umlsmatcher=matcher))
    dfref['umls'] = concepts
    dfref_m = dfref[[cid, sid, 'summary', 'umls']].groupby(cid)
    dfref_m = dfref_m.agg({
        sid: lambda x: x.to_list(),
        'summary': lambda x: x.to_list(),
        'umls': lambda x: agg_merge_umls_concept(x, majority=majority),
    }).reset_index()
    
    # get hypothesis concepts
    metrics = []
    for (hypo_file, meta_file) in files:
        dfhypo = read_meta_file(meta_file, hypo_file, cid=cid)
        assert(dfhypo['hypo'].notnull().sum() == dfhypo.shape[0])  
        if remove_duplicate:
            dfhypo = dfhypo.groupby([cid]).agg(lambda x: x.iloc[0]).reset_index()  
        concepts = []
        for j in tqdm(range(dfhypo.shape[0])):
            s = dfhypo['hypo'].iloc[j]
            concepts.append(apply_extract_umls_concepts(s, umlsmatcher=matcher, return_type='set'))
        dfhypo['umls_hypo'] = concepts
        
        # merge reference and hypothesis and calculate F-type metrics
        df = dfhypo.merge(dfref_m, on=cid, how='inner')
        assert(df.shape[0] == dfhypo.shape[0])
        metrics.append(np.vstack(df.apply(apply_naive_evaluation, axis=1)))
        
    return df, metrics
