"""Decode generated token IDs, removing ONLY terminal tokenizer EOS tokens."""
def response_text(sequence, tokenizer):
    ids=list(sequence["tokens"])
    eos=tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    while ids and ids[-1] in eos_ids:
        ids.pop()
    # Keep interior special tokens and all generated content. No fences are
    # stripped, no JSON is repaired, and no rows/numbers are changed.
    return tokenizer.decode(ids,skip_special_tokens=False)
