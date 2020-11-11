# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F

# from prepare_data import START_TOKEN, END_TOKEN, PAD_TOKEN, VOCAB, VOCAB_TOKENS, ROUND, prepare_expr_string
# from maths import Math


# ------ class ------ #
class BeamHypotheses(object):
    def __init__(self, beam_size, max_length, length_penalty, early_stopping):
        """Initialize n-best list of hypotheses."""
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beam_size = beam_size
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.beam_size:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence."""
        if len(self) < self.beam_size:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def top_k_top_p_filtering(
        logits: Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -1e10,
        min_tokens_to_keep: int = 1,
    ) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def beam_search(
        model,
        obs,
        beam_size,
        max_length,
        min_length,
        tokenizer,
        input_str = None,
        do_sample = True,
        early_stopping = False,
        temperature = 1.0,
        top_k = 10,
        top_p = 0.9,
        repetition_penalty = 1.4,
        length_penalty = 1
    ):
    """Hacker version, originally from huggingface generation utils
    https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py
    """
    
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."

    # create the first inputs for model

    if input_str is None:
        input_ids = torch.ones((batch_size * beam_size, 1)).long() * tokenizer.bos_id()
    else:
        seq = [[tokenizer.bos_id()] + tokenizer.encode(input_str)][:max_length]
        seq = [seq,]*(batch_size * beam_size)
        input_ids = torch.from_numpy(np.asarray(seq)).long().squeeze(1)

    cur_len = input_ids.size(1)
    attention_mask = torch.ones((batch_size * beam_size, cur_len)).long()
    enc_out = model.enc_out(**obs)[0]  # get the encoder output for cached

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros(
        (batch_size, beam_size),
        dtype=torch.float, device=input_ids.device
    )

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * beam_size,)

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        # (batch_size * beam_size, cur_len, vocab_size)
        logits = model.dec_out(enc_out, input_ids, attention_mask, verbose=False)[0{}]
        # (batch_size * beam_size, vocab_size)
        next_token_logits = logits[:, -1, :]

        # (batch_size * beam_size, vocab_size)
        scores = F.log_softmax(next_token_logits, dim=-1)

        #L667 --- #L62 postprocess_next_token_scores()
        if repetition_penalty != 1.0:
            for i in range(batch_size * beam_size):
                for previous_token in set(input_ids[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    repetition_penalty = repetition_penalty if scores[i, previous_token] < 0 else (1 / repetition_penalty)
                    scores[i, previous_token] *= repetition_penalty

        # set eos token prob to zero if min_length is not reached
        if eos_id is not None and cur_len < min_length:
            scores[:, eos_id] = -1e10 # -float("inf") causes nan issues

        #L108 --- #L680
        assert scores.shape == (batch_size * beam_size, vocab_size), f"Shapes of scores: {scores.shape} != {(batch_size * beam_size, vocab_size)}"

        if do_sample:
            # (batch_size * beam_size, vocab_size)
            _scores = scores + beam_scores[:, None].expand_as(scores)
            # Temperature
            if temperature != 1.0:
                _scores = _scores / temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2)  # (batch_size * beam_size, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(batch_size, beam_size * vocab_size)  # (batch_size, beam_size * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            # (batch_size, beam_size * 2)
            next_tokens = torch.multinomial(probs, num_samples=2 * beam_size)
            # Compute next scores
            # (batch_size, beam_size * 2)
            next_scores = torch.gather(_scores, -1, next_tokens)
            # sort the sampled vector to make sure that the first beam_size samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            # (batch_size, beam_size * 2)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)

        else:
            # (batch_size * beam_size, vocab_size)
            next_scores = scores + beam_scores[:, None].expand_as(scores)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(batch_size, beam_size * vocab_size) # (batch_size, beam_size * vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * beam_size)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= beam_size
                ), "Batch can only be done if at least {} beams have been generated".format(beam_size)
                assert (
                    eos_id is not None and pad_id is not None
                ), "generated beams >= beam_size -> eos_id and pad_token have to be defined"
                next_batch_beam.extend(
                    [(0, pad_id, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * beam_size + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_id is not None) and (token_id.item() == eos_id):
                    # if beam_token does not belong to top beam_size tokens, it should not be added
                    is_beam_token_worse_than_top_beam_size = beam_token_rank >= beam_size
                    if is_beam_token_worse_than_top_beam_size:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(),
                        beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == beam_size:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == beam_size, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (batch_idx + 1), "We should have added beam_size each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * beam_size
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # extend attention_mask for new generated input if only decoder
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_id is not None and all(
            (token_id % vocab_size).item() != eos_id for token_id in next_tokens[batch_idx]
        ):
            assert (
                torch.all(next_scores[batch_idx, :beam_size] == beam_scores.view(batch_size, beam_size)[batch_idx]),
                f"If batch_idx is not done, final next scores: {next_scores[:, :beam_size][batch_idx]}"
                f" have to equal to accumulated beam_scores: {beam_scores.view(batch_size, beam_size)[batch_idx]}"
            )

        # need to add best beam_size hypotheses to generated hyps
        for beam_id in range(beam_size):
            effective_beam_id = batch_idx * beam_size + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    strs = create_expr([x[1] for x in generated_hyps[0].beams])
    scores = [x[0] for x in generated_hyps[0].beams]
    return strs, scores


@torch.no_grad()
def predict_expression(
        model,
        obs,
        beam_size,
        max_length,
        min_length,
        input_str=None,
        bos_id=VOCAB[START_TOKEN],
        eos_id=VOCAB[END_TOKEN],
        pad_id=VOCAB[PAD_TOKEN],
        vocab_size=len(VOCAB),
        do_sample=True,
        early_stopping=False,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.4,
        length_penalty=1
    ):
    """wrapper for beam search
    
    :parma model: nn.Module object that the model
    :parma obs: output Encoder dict from O2fDataset
    :parma beam_size: Beam size to search on
    :parma max_length: Maximum sequence length to generate
    :parma min_length: Minimum sequence length to generate
    :parma input_str: If user has already given some input
    :parma bos_id: BOS ID
    :parma eos_id: EOS ID
    :parma pad_id: PAD ID
    :parma vocab_size: Vocabulary size
    :parma do_sample: To perform sampling or not
    :parma early_stopping: Whether to stop the beam search when at least num_beams sentences are finished per batch or not
    :parma temperature: The value used to module the next token probabilities
    :parma top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering
    :parma top_p: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or
        higher are kept for generation
    :parma repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. See `this paper
        <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    :parma length_penalty: Exponential penalty to the length. 1.0 means no penalty.
        Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
        order to encourage the model to produce longer sequences.
    """
    return beam_search(
        model=model,
        obs=obs,
        beam_size=beam_size,
        max_length=max_length,
        min_length=min_length,
        input_str=input_str,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        vocab_size=vocab_size,
        do_sample=do_sample,
        early_stopping=early_stopping,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty
    )
