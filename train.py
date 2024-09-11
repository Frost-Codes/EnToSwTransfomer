import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm
import warnings


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from decoders import greedy_decode


def get_all_sentences(ds, lang):
    for item in ds['train']:
        yield str(item[lang])


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # if tokenizer does not exist create it
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    # fetch dataset
    ds_raw = load_dataset(f"{config['datasource']}")

    # tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw['train']:
        src_ids = tokenizer_src.encode(str(item[config['src_lang']])).ids
        tgt_ids = tokenizer_tgt.encode(str(item[config['tgt_lang']])).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # split dataset 90% training 10% validation
    train_ds_size = int(0.9 * len(ds_raw['train']))
    val_ds_size = len(ds_raw['train']) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw['train'], [train_ds_size, val_ds_size])

    # create training and validation sets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],
                              config['seq_len'])

    # create data loader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config['seq_len'])
    return model


def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            # get data
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # run greedy decode
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ': >12}{source_text}")
            print_msg(f"{f'TARGET: ': >12}{target_text}")
            print_msg(f"{f'PREDICTED: ': >12}{model_out_text}")

            if count == num_examples:
                print_msg('=' * console_width)
                break


def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    if str(device) == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    else:
        print("Using cpu. If you have a GPU, consider using it for training.")

    # weights folder
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # load dataset and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # preload model if specified before training
    initial_epoch = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
    else:
        print('No model to preload, starting from scratch')

    # start training
    for epoch in range(initial_epoch + 1, config['num_epochs'] + 1):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # forward pass run input through encoder, decoder and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)

            # move labels to device
            label = batch['label'].to(device)  # (B, seq_len)

            # calculate loss
            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item(): 6.3f}"})

            # backward
            loss.backward()

            # update the weights
            optimizer.step()

            # zero out grad
            optimizer.zero_grad(set_to_none=True)

        # run validation after each epoch
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg))

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch: 02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)














