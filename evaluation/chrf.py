import sacrebleu

def compute_chrf(predictions, references):
    """
    Computes chrF++ score for translation lists.
    
    Args:
        predictions (list[str]): Generated English sentences.
        references (list[str]): List of reference sentences.
        
    Returns:
        float: chrF++ score.
    """
    score = sacrebleu.corpus_chrf(predictions, [references], word_order=2) # word_order=2 makes it chrF++
    return score.score
