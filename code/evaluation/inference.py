#from fairseq.models.bart import BARTModel
import argparse
import inspect
import torch
from rouge import Rouge, FilesRouge
import os
import numpy as np
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import defaultdict
from fairseq.models.bart import BARTModel
from tqdm import tqdm as tqdm
#from rouge import Rouge
from rouge_score import rouge_scorer


#%% Helper Functions
def apply_rouge_scores_calculation(value, scorer):
    # helper functions for calculating rouge scores
    refs = value['ref']
    hypos = [value['hypo']]*len(refs)
    scores = defaultdict(lambda: defaultdict(list))
    for (h, r) in zip(hypos, refs):
        s = scorer.score(r, h)
        for key in s:
            scores[key]['p'].append(s[key].precision)
            scores[key]['r'].append(s[key].recall)
            scores[key]['f'].append(s[key].fmeasure)
    return pd.DataFrame(scores).applymap(np.mean)


def calculate_rouge_scores(df, rougescorer):
    # helper function for applying rouge score calculation to dataframe object
    df['scores'] = df.apply(apply_rouge_scores_calculation, scorer=rougescorer, axis=1)
    return (sum(df['scores'])/df.shape[0]).to_dict()


def calculate_rouge_simple(hypos, refs, scorer):
    # helper functions for calculating rouge scores
    scores = defaultdict(lambda: defaultdict(list))
    for (h, r) in zip(hypos, refs):
        s = scorer.score(r, h)
        for key in s:
            scores[key]['p'].append(s[key].precision)
            scores[key]['r'].append(s[key].recall)
            scores[key]['f'].append(s[key].fmeasure)
    return pd.DataFrame(scores).applymap(np.mean).to_dict()
    

def inference_cached(bart, source_file, batch_size=24, max_len=512, nbeam=4):
    # helper function: inference with single model checkpoint with caching feature
    bart.eval()
    cache = {}
    with open(source_file) as source:
        lines = source.read().strip().split('\n')
        pbar = tqdm(total=len(lines))  # manually control progress bar for while loop usage
        i = 0
        slines = []
        while i < len(lines):
            if lines[i] not in cache and lines[i] not in slines:
                slines.append(lines[i])
                if len(slines) >= batch_size:
                    with torch.no_grad():
                        hypotheses_batch = bart.sample(slines, beam=nbeam, lenpen=2.0, max_len_b=max_len, min_len=5, no_repeat_ngram_size=3)
                        for (src, hypo) in zip(slines, hypotheses_batch):
                            cache[src] = hypo
                        pbar.update(i)
                    slines.clear()
            i += 1
        if len(slines) > 0:
            # possible remaining sentences that don't fill up a batch
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=nbeam, lenpen=2.0, max_len_b=max_len, min_len=5, no_repeat_ngram_size=3)
                for (src, hypo) in zip(slines, hypotheses_batch):
                    cache[src] = hypo
                pbar.update(i)
        pbar.close()
    hypotheses = [cache[x] for x in lines] 
    return hypotheses
            
    
def validate_checkpoint_filename(s):
    # helper function for checking filename validity of model checkpoints
    status = True
    status = status and s.endswith('.pt')
    status = status and s.startswith('checkpoint')
    status = status and s.replace('checkpoint', '').replace('.pt', '').replace('_', '').isdigit()
    return status


#%% Main API
def batch_inference(dataset, bin_path, datafolder,
                    cid='cid',
                    suffix='', 
                    refsuffix='', 
                    ckptfolder='checkpoints/', 
                    ckpt_file=None, 
                    calculate_rouge=True, 
                    calculation_type='old',
                    **kwargs):
    """Run batch model inference on a dataset.
    
    Required Parameters
    -------------------
    dataset: str, name prefix of the dataset files
    bin_path: str, path to binarized data folder used in model training
    datafolder: str, path to folder containing the dataset

    Keyword Parameters
    ------------------
    cid: str (default 'cid'), name of the column used for conversation identifier
    suffix: str (default ''), name suffix of the dataset files
    refsuffix: str (default ''), name suffix of the reference files
    ckptfolder: str (default 'checkpoints/'), path to the model checkpoint folder
    ckpt_file: str (default: None), if provided, uses only specified model checkpoint instead of iterating over all model checkpoints in <ckptfolder>
    calculate_rouge: bool (default: True), whether to calculate rouge scores
    calculation_type: str (default: 'old'), deprecated, no need to change
    **kwargs: additional keyword parameters supported by inference_cached() API

    Return
    ------
    best_ckpt: str, the file name of the best performing model checkpoint
    """
    assert(calculation_type in ['old', 'new'])
    # run batch inference using checkpoints in ckptfolder, checkpoint file must be in the format "checkpoint[0-9]+.pt"
    rougescorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=False)
    
    source_file = os.path.join(datafolder, f'{dataset}{suffix}.source')
    source_meta_file = os.path.join(datafolder, f'{dataset}{suffix}.meta')
    ref_orig_file = os.path.join(datafolder, f'{dataset}{suffix}.target')
    ref_file = os.path.join(datafolder, f'{dataset}{refsuffix}.target')
    ref_meta_file = os.path.join(datafolder, f'{dataset}{refsuffix}.meta')
    hypo_file = os.path.join(datafolder, '{}{}.hypo'.format(dataset, suffix))
    
    if calculate_rouge:
        if calculation_type == 'new':
            ref = pd.read_csv(ref_meta_file, sep='\t', index_col=0, header=0)
            ref['ref'] = open(ref_file, 'r').read().strip().split('\n')
            assert(ref['ref'].notnull().sum() == ref.shape[0]) # make sure the # of examples agree between meta and target files
            ref = ref[[cid, 'ref']].groupby(cid).agg(lambda x: x.to_list())
            hyp = pd.read_csv(source_meta_file, sep='\t', index_col=0, header=0)
            hyp = hyp[[cid]]
        else:
            ref = open(ref_orig_file).read().strip().split('\n')
    
    r1f_old = 0
    r2f_old = 0
    rlf_old = 0
    best_ckpt = ''
    
    if ckpt_file is None:
        flist = sorted([x for x in os.listdir(ckptfolder) if validate_checkpoint_filename(x)], reverse=False)
    elif type(ckpt_file) is str:
        flist = [ckpt_file]
    elif type(ckpt_file) is list:
        flist = sorted(ckpt_file, reverse=True)
    for f in flist:
        print('='*50)
        print('loading {}'.format(f))
        
        bartmodel = BARTModel.from_pretrained(
            ckptfolder,
            checkpoint_file=f,
            data_name_or_path=bin_path
        )

        print('starting inference')
        print('-'*50)
        
        if torch.cuda.is_available():
            bartmodel.cuda()
            
        generated = inference_cached(bartmodel, source_file, **kwargs)

        del bartmodel
        torch.cuda.empty_cache()
        
        if calculate_rouge: 
            if calculation_type == 'new':
                hyp['hypo'] = generated
                final = hyp.merge(ref, on=cid, how='left')
                scores = calculate_rouge_scores(final, rougescorer)
            else:
                assert(len(generated) == len(ref))
                scores = calculate_rouge_simple(generated, ref, rougescorer)
            
            print(f'{dataset}', scores)
            r1f = scores['rouge1']['f']
            r2f = scores['rouge2']['f']
            rlf = scores['rougeL']['f']
            check = (r1f > r1f_old) + (r2f > r2f_old) + (rlf > rlf_old)
            if check >= 2:
                with open(hypo_file, 'w', encoding='utf-8') as fout:
                    for h in generated:
                        fout.write(h + '\n')
                        fout.flush()
                r1f_old = r1f
                r2f_old = r2f
                rlf_old = rlf
                best_ckpt = f
                print('-'*50)
                if len(flist) > 1:
                    print(f'better results obtained, saving hypotheses to {hypo_file}')
                else:
                    print(f'inference done, saving hypotheses to {hypo_file}')
        else:
            with open(hypo_file, 'w', encoding='utf-8') as fout:
                for h in generated:
                    fout.write(h + '\n')
                    fout.flush()
            print('Inference done, results saved to {}'.format(hypo_file))
            
    if len(flist) > 1:
        print('Best checkpoint: {}'.format(best_ckpt))
    
    return best_ckpt


if __name__ == '__main__':
    # parsing cli
    parser = argparse.ArgumentParser(description='Bart summarizer evaluation script')
    
    parser.add_argument('--bin_path', type=str, default=None,
                        help='name of the binarized data folder used for training the summarizer; if not specified, infer from train.sh')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints/',
                        help='path to checkpoint folders')
    parser.add_argument('--best_checkpoint', type=str,  default=None,
                        help='name of the checkpoint file')
    parser.add_argument('--data_folder', type=str,  default='../experiments',
                        help='path to experiment folder')
    parser.add_argument('--conv_id', type=str,  default='cid',
                        help='unique identifier for conversations')
    parser.add_argument('--train_suffix', type=str,  default=None,
                        help='suffix for dev set source/target files')
    parser.add_argument('--dev_suffix', type=str,  default='',
                        help='suffix for dev set source/target files')
    parser.add_argument('--test_suffix', type=str,  default='',
                        help='suffix for test set source/target files')
    parser.add_argument('--ref_suffix', type=str,  default='_all',
                        help='suffix for target files containing all target summaries, used in rouge_against_all calculation')
    parser.add_argument('--max_len', type=int,  default=512,
                        help='maximum token limit')
    parser.add_argument('--nbeam', type=int,  default=4,
                        help='no. of beams')
    parser.add_argument('--batch_size', type=int,  default=12,
                        help='batch size')
    parser.add_argument('--no_rouge', action='store_true',
                        help='whether to skip rouge calculation (target file no longer needed)')
    parser.add_argument('--rouge_against_all', action='store_true',
                        help='whether to calculate rouge score against all references')
    parser.add_argument('--skip_dev', action='store_true',
                        help='whether to skip inference on dev set')
    parser.add_argument('--skip_test', action='store_true',
                        help='whether to skip inference on test set')
    
    args = parser.parse_args()
    
    if args.bin_path is None:
        with open('../train.sh') as reader:
            lines = reader.read().split('\n')
            mdl = [x for x in lines if x.startswith('BART_PATH=')][0]
            mdl = mdl.split('/')[-1].split('.')[0]
            expname = [x for x in lines if x.startswith('EXP=')][0]
            bin_path = os.path.join(args.data_folder, expname[4:].strip(), 'bin')
    else:
        bin_path = args.bin_path
        mdl = 'model'
    
    ckpt_file = args.best_checkpoint
    ckptfolder = args.checkpoint_folder
    datafolder = args.data_folder
    
    cid = args.conv_id
    trainsuffix = args.train_suffix
    devsuffix = args.dev_suffix
    testsuffix = args.test_suffix
    refsuffix = args.ref_suffix
    
    # get keyword arguments used in inference() from command line inputs
    sig = inspect.getfullargspec(inference)
    kwds = {x: getattr(args, x) for x in sig.args[-len(sig.defaults):]}
    calculate_rouge = not args.no_rouge
    calculation_type = 'new' if args.rouge_against_all else 'old'
    
    print('='*50)
    print(f'Datapath = {bin_path}')
    print(f'Checkpoint folder = {ckptfolder}')
    print(f'Checkpoint file = {ckpt_file}')
    print('-'*50)
    print(f'Data folder = {datafolder}')
    print(f'Train file suffix = {trainsuffix}')
    print(f'Dev file suffix = {devsuffix}')
    print(f'Test file suffix = {testsuffix}')
    print(f'Conv ID column name = {cid}')
    print(f'Ref file suffix (with all summaries) = {refsuffix}')
    print('-'*50)
    print('Beam size = {}'.format(kwds['nbeam']))
    print('Maximum token = {}'.format(kwds['max_len']))
    print('Batch size = {}'.format(kwds['batch_size']))
    print(f'Calculate Rouge? {calculate_rouge}')
    if calculate_rouge:
        print(f'Calculate Rouge against all references? {args.rouge_against_all}')
    print(f'Skip dev set? {args.skip_dev}')
    print(f'Skip test set? {args.skip_test}')
    print('='*50)
    
    if not args.skip_dev:
        best_ckpt = batch_inference(
            'dev', bin_path, datafolder,
            cid=cid,
            suffix=devsuffix,
            refsuffix=refsuffix,
            ckptfolder=ckptfolder, 
            ckpt_file=ckpt_file, 
            calculate_rouge=calculate_rouge, 
            calculation_type=calculation_type, 
            **kwds
        )
    
        if ckpt_file is None:
            ckpt_file = best_ckpt
            # also save best checkpoint results to log file
            log_file = os.path.join(datafolder, 'best_model_checkpoint{}.log'.format(devsuffix))
            w = 'w'
            with open(log_file, mode=w) as writer:
                writer.write('====Best Model Checkpoint====\n{}\n'.format(ckpt_file))
    
    if not args.skip_test:
        batch_inference(
            'test', bin_path, datafolder, 
            suffix=testsuffix,
            refsuffix=refsuffix,
            ckptfolder=ckptfolder, 
            ckpt_file=ckpt_file, 
            calculate_rouge=calculate_rouge, 
            calculation_type=calculation_type, 
            **kwds
        )
    
    if trainsuffix is not None:
        batch_inference(
            'train', bin_path, datafolder, 
            suffix=trainsuffix,
            refsuffix=refsuffix,
            ckptfolder=ckptfolder, 
            ckpt_file=ckpt_file, 
            calculate_rouge=calculate_rouge, 
            calculation_type=calculation_type, 
            **kwds
        )
    
    print('Best checkpoint is: ', ckpt_file)