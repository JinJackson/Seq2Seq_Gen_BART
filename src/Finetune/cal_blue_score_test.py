import sys

# sys.path.append("../")
# from utils import bleu_score
from nlgeval import compute_metrics


# print(bleu_score)
ref_file = "test1.txt"
gen_file = 'test2.txt'

# res_origin = bleu_score(ref_file, gen_file)
metrics_dict = compute_metrics(hypothesis=gen_file, references=[ref_file,])
# print(res_origin)
print(metrics_dict)