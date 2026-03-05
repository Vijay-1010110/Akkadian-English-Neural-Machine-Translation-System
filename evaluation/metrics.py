import math
from evaluation.bleu import compute_bleu
from evaluation.chrf import compute_chrf

def evaluate_metrics(predictions, references):
    """
    Computes BLEU, chrF++, and the Kaggle Competition Metric:
    Score = sqrt(BLEU * chrF++)
    """
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    
    # Competition metric
    comp_score = math.sqrt(bleu * chrf) if bleu > 0 and chrf > 0 else 0.0
    
    return {
        "bleu": bleu,
        "chrf": chrf,
        "competition_score": comp_score
    }
