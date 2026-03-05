import sacrebleu

def compute_bleu(predictions, references):
    """
    Computes SacreBLEU score for translation lists.
    
    Args:
        predictions (list[str]): Generated English sentences.
        references (list[str]): List of reference sentences.
        
    Returns:
        float: BLEU score.
    """
    # SacreBLEU expects a list of lists for references if multiple references are used
    # For single references, wrap the whole list in another list
    score = sacrebleu.corpus_bleu(predictions, [references])
    return score.score
