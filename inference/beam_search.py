import torch
import logging

logger = logging.getLogger(__name__)

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, pad_symbol, beam_size=5, length_penalty=1.0, repetition_penalty=1.2):
    """
    Implements standard Beam Search for custom models lacking `.generate()`.
    Adapted for Transformer architecture.
    """
    device = src.device
    batch_size = src.size(0)
    
    # Standard encode - memory shape [batch, seq, d_model]
    memory, memory_padding_mask = model.encode(src, src_key_padding_mask=src_mask)
    
    # Shape tracking: [batch_size, beam_size, seq_len]
    # We maintain a list of completed hypotheses per batch item
    completed_hyps = [[] for _ in range(batch_size)]
    
    # Starting tokens: [batch_size, 1, 1]
    running_seqs = torch.full((batch_size, 1, 1), start_symbol, dtype=torch.long, device=device)
    running_scores = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
    
    # We iteratively expand building the beam
    for step in range(max_len):
        current_seq_len = running_seqs.size(-1)
        
        # We need to reshape for the forward pass if beam_size > 1
        # [batch_size * current_beam_size, seq_len]
        flat_seqs = running_seqs.view(-1, current_seq_len)
        flat_memory = memory.repeat_interleave(running_seqs.size(1), dim=0)
        flat_memory_padding_mask = memory_padding_mask.repeat_interleave(running_seqs.size(1), dim=0) if memory_padding_mask is not None else None
        
        # Decode: [batch * current_beam, seq, vocab]
        logits = model.decode(flat_seqs, flat_memory, memory_key_padding_mask=flat_memory_padding_mask)
        
        # Get log probs of the *last* predicted token
        next_token_logits = logits[:, -1, :] # [batch * current_beam, vocab]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(next_token_logits.size(0)):
                # Penalty logic: if score < 0, multiply by penalty. If > 0, divide.
                # Find which tokens have already been generated in this beam
                for token_id in set(flat_seqs[i].tolist()):
                    if next_token_logits[i, token_id] < 0:
                        next_token_logits[i, token_id] *= repetition_penalty
                    else:
                        next_token_logits[i, token_id] /= repetition_penalty
                        
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        
        # Expand step
        vocab_size = next_token_logprobs.size(-1)
        current_beam_size = running_seqs.size(1)
        
        # Reshape to easily add to running scores
        # [batch_size, current_beam_size, vocab_size]
        next_token_logprobs = next_token_logprobs.view(batch_size, current_beam_size, vocab_size)
        
        # Add running scores: [batch_size, current_beam_size, 1] + [batch, current_beam, vocab]
        cumulative_scores = running_scores.unsqueeze(-1) + next_token_logprobs
        
        # [batch_size, current_beam_size * vocab_size]
        cumulative_scores = cumulative_scores.view(batch_size, -1)
        
        # Top-K
        k_to_keep = min(beam_size, cumulative_scores.size(1))
        topk_scores, topk_indices = torch.topk(cumulative_scores, k_to_keep, dim=-1)
        
        # Which beam did it come from, and what was the actual token?
        beam_indices = torch.div(topk_indices, vocab_size, rounding_mode='floor')
        token_indices = topk_indices % vocab_size
        
        # Build the new sequences
        new_running_seqs = []
        new_running_scores = []
        
        for b in range(batch_size):
            b_new_seqs = []
            b_new_scores = []
            
            for i in range(k_to_keep):
                beam_idx = beam_indices[b, i]
                token = token_indices[b, i]
                score = topk_scores[b, i]
                
                prev_seq = running_seqs[b, beam_idx]
                new_seq = torch.cat([prev_seq, token.unsqueeze(0)])
                
                if token == end_symbol:
                    # Score penalization for sequence length
                    final_score = score / (len(new_seq) ** length_penalty)
                    completed_hyps[b].append((new_seq, final_score))
                else:
                    b_new_seqs.append(new_seq)
                    b_new_scores.append(score)
            
            # If we don't have enough running beams to continue (all finished), pad it out or stop
            if not b_new_seqs:
                # Add a dummy pad so shapes match
                b_new_seqs = [torch.full((current_seq_len + 1,), pad_symbol, dtype=torch.long, device=device)]
                b_new_scores = [torch.tensor(float('-inf'), device=device)]
                
            # Keep only up to beam_size running sequences
            new_running_seqs.append(torch.stack(b_new_seqs[:beam_size]))
            new_running_scores.append(torch.stack(b_new_scores[:beam_size]))
            
        running_seqs = torch.stack(new_running_seqs) # [batch_size, new_beam_size, curr_len + 1]
        running_scores = torch.stack(new_running_scores) # [batch_size, new_beam_size]
        
    # Gather best hypothesis per batch
    best_seqs = []
    for b in range(batch_size):
        if not completed_hyps[b]:
            # No sequence reached EOS, just take the running sequence with best score
            best_idx = torch.argmax(running_scores[b])
            best_seq = running_seqs[b, best_idx]
        else:
            # Sort completed hypotheses by score descending and take the top
            completed_hyps[b].sort(key=lambda x: x[1], reverse=True)
            best_seq = completed_hyps[b][0][0]
            
        best_seqs.append(best_seq)
        
    return best_seqs
