import sys
import json


def parse(tokens):
    if "(" not in tokens:
        assert ")" not in tokens
        ret = dict()
        start = 0
        mid = 0
        for ii, tok in enumerate(tokens):
            if tok == "«":
                mid = ii
            elif tok == "»":
                key = ' '.join(tokens[start:mid])
                val = ' '.join(tokens[mid + 1:ii])
                ret[key] = val
                start = mid = ii + 1
        return ret

    st = tokens.index("(")
    outer_key = ' '.join(tokens[0:st])
    assert tokens[-1] == ")", " ".join(tokens)

    level = 0
    last = st + 1
    ret = dict()
    for ii in range(st + 1, len(tokens) - 1, 1):
        tok = tokens[ii]
        if tok == "»" and level == 0:
            rr = parse(tokens[last:ii + 1])
            ret.update(rr)
            last = ii + 1
        elif tok == "(":
            level += 1
        elif tok == ")":
            level -= 1
            if level == 0:
                rr = parse(tokens[last:ii + 1])
                ret.update(rr)
                last = ii + 1

    return {outer_key: ret}


def load_jsonl(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as fp:
        for line in fp:
            data.append(json.loads(line.strip()))

    return data


def per_sample_metric(gold, pred):
    ret = dict()
    ret['accuracy'] = int(gold == pred)

    get_intent = lambda x: x.split('(', 1)[0].strip()
    gintent = get_intent(gold)
    pintent = get_intent(pred)
    ret['intent_accuracy'] = int(gintent == pintent)

    parse_correct = 1
    try:
        _ = parse(pred.split())
    except:
        parse_correct = 0
    ret['parsing_accuracy'] = parse_correct

    return ret


def compute_metrics(data, preds):
    assert len(data) == len(preds), "Different number of samples in data and prediction."

    golds = [x['output'] for x in data]

    metrics = [per_sample_metric(gold, pred) for gold, pred in zip(golds, preds)]
    final_metrics = dict()
    mnames = list(metrics[0].keys())
    for key in mnames:
        final_metrics[key] = sum([met[key] for met in metrics]) / len(golds)
    
    return final_metrics


class Evaluator():
    def __init__(self, data_file):
        self.data = load_jsonl(data_file)

    def compute_metrics(self, preds):
        metrics = compute_metrics(self.data, preds)
        return metrics
