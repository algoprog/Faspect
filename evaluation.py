import itertools

from bert_score import score
from nltk.translate.bleu_score import sentence_bleu


def best_bleu_cand(groundtruth, candidate):
    assert len(groundtruth) >= len(candidate)
    all_permutations = list(itertools.permutations(candidate))
    max_bleu = 0.
    best_cand = all_permutations[0]
    for cand in all_permutations:
        bleu = 0.
        for i in range(min(len(groundtruth), len(cand))):
            bleu += sentence_bleu([groundtruth[i]], cand[i]) / len(groundtruth)
        if bleu > max_bleu:
            max_bleu = bleu
            best_cand = cand
    return list(best_cand)


def eval_bleu(groundtruth, cand):
    # Calculates the SET BLEU metrics, for 1-gram, 2-gram, 3-gram and 4-gram overlaps
    best_cand = best_bleu_cand(groundtruth, cand)
    bleu = [0., 0., 0., 0.]
    bleu_weights = [[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0.33, 0.33, 0.33, 0], [0.25, 0.25, 0.25, 0.25]]
    for j in range(4):
        for i in range(min(len(groundtruth), len(best_cand))):
            bleu[j] += sentence_bleu([groundtruth[i]], best_cand[i], weights=bleu_weights[j]) / len(groundtruth)
    return bleu


def bertscore(groundtruth, cand):
    # Calculates the Set BERT-Score metrics for Precision, Recall & F1
    best_cand = best_bleu_cand(groundtruth, cand)
    (P, R, F), hashname = score(best_cand, groundtruth, lang="en", return_hash=True, device="cuda:0")
    return P.mean().item(), R.mean().item(), F.mean().item()


def exact_match(groundtruth, cand):
    # Calculates the exact match Precision, Recall & F1
    c = 0.
    for x in cand:
        if x != '' and x in groundtruth:
            c += 1
    p = c / (len([x for x in cand if x != ''])+1e-8)
    r = c / (len([x for x in groundtruth if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]


def term_match(groundtruth, cand):
    # Calculates the term overlap Precision, Recall & F1
    gt_terms = set([])
    for x in groundtruth:
        if x == '':
            continue
        for t in x.strip().split():
            gt_terms.add(t)
    cand_terms = set([])
    for x in cand:
        if x == '':
            continue
        for t in x.strip().split():
            cand_terms.add(t)

    c = 0.
    for x in cand_terms:
        if x != '' and x in gt_terms:
            c += 1
    p = c / (len([x for x in cand_terms if x != ''])+1e-8)
    r = c / (len([x for x in gt_terms if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]


if __name__ == "__main__":
    groundtruth = ["for sale", "used cars", "electric", "cheap"]
    cand = ["afforable cars", "cars for sale", "used", "electric"]

    term_overlap_metrics = term_match(groundtruth, cand)
    print("Term overlap metrics: P={},R={},F1={}".format(term_overlap_metrics[0],
                                                         term_overlap_metrics[1],
                                                         term_overlap_metrics[2]))

    exact_match_metrics = exact_match(groundtruth, cand)
    print("Exact match metrics: P={},R={},F1={}".format(exact_match_metrics[0],
                                                        exact_match_metrics[1],
                                                        exact_match_metrics[2]))

    bert_score_metrics = bertscore(groundtruth, cand)
    print("BERT score metrics: P={},R={},F1={}".format(bert_score_metrics[0],
                                                       bert_score_metrics[1],
                                                       bert_score_metrics[2]))
