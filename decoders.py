import torch


from dataset import causal_mask


def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # compute encoder output and use it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)

    # initialize decoder input <sos> token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

    # initialize loop, we will generate tokens until max_len or we encounter </eos>
    while True:
        if decoder_input.size(1) == max_len:  # case 1
            break
        # decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # calculate output
        out = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # run projection layer
        proj = model.project(out[:, -1])

        # find next word (greedy approach)
        _, next_word = torch.max(proj, dim=1)

        # create new decoder input
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:  # case 2
            break

    return decoder_input.squeeze(0)




