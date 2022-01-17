import warnings

warnings.filterwarnings('ignore')

import hparams
import transformers
import os
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataset import MyDataset
from TransformerTTSModel import LSTransformer, PostNet
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from collections import OrderedDict
import plot
import numpy as np
import matplotlib.pyplot as plt
import audio


def load_data_to_dataloader(dataset, num_workers=hparams.num_workers, shuffle=True):
    return DataLoader(dataset,
                      batch_size=hparams.batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=False)


def load_checkpoint(step, model_name="transformer"):
    state_dict = torch.load('./logs/checkpoint/checkpoint_%s_%d.pth.tar' % (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[:]
        new_state_dict[key] = value

    return new_state_dict


def loss_fn(
    target_mel_spect,
    pred_mel_spect_post,
    pred_mel_spect,
    ):
    mel_loss = nn.L1Loss()(pred_mel_spect, target_mel_spect) + nn.L1Loss()(pred_mel_spect_post, target_mel_spect)
    return mel_loss


def train(model, loader, optimizer, writer, epoch, data_len):
    global attn_probs, attns_enc, attn_dec, mel_postpred, mel, mel_pred
    running_loss = 0
    model.train()
    count = 0
    training_loss = []
    encoder_alpha = []
    decoder_alpha = []
    for (i_iter, input_data) in tqdm(enumerate(loader), total=data_len/hparams.batch_size):
        for p in model.parameters():
            p.grad = None

        vid = input_data[0].cuda()
        mel = input_data[1].cuda()
        mel_mask = input_data[2].cuda()
        end_logits = input_data[3].cuda()
        start_token = input_data[4].cuda()
        end_token = input_data[5].cuda()
        empty_token = input_data[6].cuda()
        empty_token = empty_token.unsqueeze(-1).repeat(1, 1, 80)
        start_token = start_token.unsqueeze(-1).repeat(1, 1, 80)
        end_token = end_token.unsqueeze(-1).repeat(1, 1, 80)

        mel_input = torch.cat((start_token, mel[:, :-1]), 1)

        # mel = torch.cat((mel, end_token), 1)
        # print("mel inp", mel_input.shape)
        mel_postpred, mel_pred, end_logits_pred, attn_dec, attn_probs, attns_enc = model(vid, mel_input, mel_mask[:, :-1])
        # print("mel pred", mel_pred.shape)
        loss = loss_fn(
            torch.cat((start_token, mel[:, 1:]), 1),
            mel_postpred,
            mel_pred,
        )
        running_loss += loss.item()
        encoder_alpha.append(model.encoder.alpha.data)
        decoder_alpha.append(model.decoder.alpha.data)
        training_loss.append(loss.detach().cpu())
        writer.add_scalars('training_loss', {
            'loss': loss,
            'running_loss': running_loss,

        }, (i_iter + 1) + ((epoch - 1) * 10))
        
        # Reset the gradient back to so it is ready for the next batch
        optimizer.zero_grad()
        loss.backward() # Update weights
        nn.utils.clip_grad_norm_ # The norm is computed over all gradients together, as if they were concatenated into a single vector.
        optimizer.step() # for each parameter p do p -= p.grad * lr
       
        count += 1
        if count == hparams.image_step:
            break
    # attention_heads_dec = list()
    # for id, attentionheads in enumerate(attn_dec):
    #     attention_heads_dec.append(
    #         attentionheads.contiguous().view(-1, 8, 4, attentionheads.shape[1], attentionheads.shape[2])[0])
    # fig, ax = plt.subplots(6, 4, figsize=(4 * 3, 6 * 3))
    # for i, attention_heads in enumerate(attention_heads_dec):
    #     # print("attention heads", attention_heads.shape)
    #     attn_maps = attention_heads[0]
    #     attn_maps = attn_maps.detach().cpu()
    #     for j in range(4):
    #         ax[i][j].imshow(attn_maps[j], origin='lower', vmin=0)
    #         ax[i][j].set_title("Layer %i, Head %i" % (i + 1, j + 1))
    #
    #         # print("atten heads[head]", attention_heads[j].shape)
    #         # ax = plt.gca()
    #         # ax.matshow(attention_heads[j])
    #     fig.subplots_adjust(hspace=0.5)
    #     plt.show()
    #
    #     plt.savefig(
    #         'D:\\Final-sem-project\\Transformer-LST\\logs\\plots\\{}_epoch_eval_alignments_layer.png'.format(
    #             epoch))
    for i, prob in enumerate(attn_probs):  # j values: 0, 1, 2, 3 ; batchsize: 2 ; prob[]: 0, 2, 4, 6
        num_h = prob.size(0)
        # prob = prob.contiguous().view(-1, prob.shape[0] * prob.shape[1], prob.shape[2], prob.shape[3])[0]
        # print("attn_prob", prob.shape)
        for j in range(4):  # Since no. of heads in decoder attention is 4
            x = vutils.make_grid(prob[j * hparams.batch_size] * 255)
            writer.add_image('Attention_%d_0' % epoch, x, i * hparams.nhead + j)

    for i, prob in enumerate(attns_enc):
        # print("attn enc", prob.shape)
        num_h = prob.size(0)
        for j in range(4):
            x = vutils.make_grid(prob[j] * 255)
            writer.add_image('Attention_enc_%d_0' % epoch, x, i * hparams.nhead + j)

    for i, prob in enumerate(attn_dec):
        num_h = prob.size(0)
        # print("i, attn_dec", i, prob.shape)
        for j in range(4):  # Since no. of heads in decoder attention is 4
            # print(j * 4, prob[j * 4].shape)
            x = vutils.make_grid(prob[j * 4] * 255)
            # y = vutils.make_grid(prob[j * hparams.batch_size])
            writer.add_image('Attention_dec_%d_0' % epoch, x, i * hparams.nhead + j)
    
    writer.flush()
    # y = y.permute(1, 2, 0).detach().cpu()
    # plt.imshow(y)
    # plt.savefig(
    #         'D:\\Final-sem-project\\Transformer-LST\\logs\\plots\\{}_epoch_eval_alignments_layer.png'.format(
    #             epoch))



    epoch_loss = running_loss
    target = np.asarray(mel[0].permute(1, 0).detach().cpu())
    postnet_pred = np.asarray(mel_postpred[0].permute(1, 0).detach().cpu())
    pred = np.asarray(mel_pred[0].permute(1, 0).detach().cpu())
    plot.plot_spectrogram(postnet_pred,
                          os.path.join(hparams.mel_path, "step-{}-mel-spectrogram.png".format(epoch)),
                          title="{}, epoch={}, loss={:.5f}".format("Transformer",
                                                                   epoch, epoch_loss),
                          target_spectrogram=target,
                          max_len=None)

    plot.plot_spectrogram(pred,
                          os.path.join(hparams.mel_path,
                                       "step-{}-decoder_output-mel-spectrogram.png".format(epoch)),
                          title="{}, epoch={}, loss={:.5f}".format("Transformer",
                                                                   epoch, epoch_loss),
                          target_spectrogram=target,
                          max_len=None)

    wav = audio.inv_mel_spectrogram(postnet_pred)
    audio.save_wav(wav,
                   os.path.join(hparams.wav_path, "step-{}-wave-from-mel.wav".format(epoch)),
                   sr=hparams.sample_rate)

    # # ===============================================================================
    # model.eval()
    # e_running_loss = 0
    # e_count = 0
    # eval_loss = []
    # with torch.no_grad():
    #     for (i_iter, input_data) in tqdm(enumerate(loader), total=data_len/hparams.batch_size):

    #         e_vid = input_data[0].cuda()
    #         e_mel = input_data[1].cuda()
    #         e_mel_mask = input_data[2].cuda()
    #         e_end_logits = input_data[3].cuda()
    #         e_start_token = input_data[4].cuda()
    #         e_end_token = input_data[5].cuda()
    #         e_start_token = e_start_token.unsqueeze(-1).repeat(1, 1, 80)
    #         e_end_token = e_end_token.unsqueeze(-1).repeat(1, 1, 80)
    #         empty_token = input_data[6].cuda()
    #         empty_token = empty_token.unsqueeze(-1).repeat(1, 1, 80)
    #         # e_mel = torch.cat((e_mel, e_end_token), 1)
    #         e_mel_inp = e_start_token
    #         for i in range(1, len(e_mel_mask[1])):
    #           e_mel_postpred, e_mel_pred, e_end_logits_pred, e_attn_dec, e_attn_probs, e_attns_enc = model(e_vid, e_mel_inp)
    #           e_mel_inp = torch.cat((e_mel_inp, e_mel_postpred[:, -1:, :]), dim=1)

    #         e_loss = loss_fn(
    #             e_mel[:, 1:],
    #             end_logits,
    #             e_mel_postpred,
    #             e_mel_pred,
    #             e_end_logits_pred
    #         )
    #         e_running_loss += e_loss.item()

    #         e_count += 1
    #         if e_count == 1:
    #             break

        
    #     e_epoch_loss = e_running_loss

    #     e_target = np.asarray(e_mel[0].permute(1, 0).detach().cpu())
    #     e_postnet_pred = np.asarray(e_mel_postpred[0].permute(1, 0).detach().cpu())
    #     e_pred = np.asarray(e_mel_pred[0].permute(1, 0).detach().cpu())
    #     plot.plot_spectrogram(e_postnet_pred,
    #                           os.path.join(hparams.mel_path, "step-{}-training-eval-mel-spectrogram.png".format(epoch)),
    #                           title="{}, epoch={}, loss={:.5f}".format("Transformer",
    #                                                                    epoch, e_epoch_loss),
    #                           target_spectrogram=e_target,
    #                           max_len=None)

    #     plot.plot_spectrogram(e_pred,
    #                           os.path.join(hparams.mel_path,
    #                                        "step-{}-training-eval-decoder_output-mel-spectrogram.png".format(epoch)),
    #                           title="{}, epoch={}, loss={:.5f}".format("Transformer",
    #                                                                    epoch, e_epoch_loss),
    #                           target_spectrogram=e_target,
    #                           max_len=None)
    #     wav = audio.inv_mel_spectrogram(e_postnet_pred)
    #     audio.save_wav(wav,
    #                    os.path.join(hparams.wav_path, "step-{}-wave-from-training-eval-mel.wav".format(epoch)),
    #                    sr=hparams.sample_rate)
            
    # model.train()

    return epoch_loss, training_loss, encoder_alpha, decoder_alpha


def evaluate(model, loader, optimizer, writer, epoch, data_len):
    e_running_loss = 0
    postnet = PostNet(mel_dims=80, hidden_dims=384, dropout=0.1).cuda()
    model.eval()
    e_count = 0
    eval_loss = []
    with torch.no_grad():
        for (i_iter, input_data) in tqdm(enumerate(loader), total=data_len/hparams.batch_size):

            e_vid = input_data[0].cuda()
            e_mel = input_data[1].cuda()
            e_mel_mask = input_data[2].cuda()
            e_end_logits = input_data[3].cuda()
            e_start_token = input_data[4].cuda()
            e_end_token = input_data[5].cuda()
            e_start_token = e_start_token.unsqueeze(-1).repeat(1, 1, 80)
            e_end_token = e_end_token.unsqueeze(-1).repeat(1, 1, 80)
            empty_token = input_data[6].cuda()
            empty_token = empty_token.unsqueeze(-1).repeat(1, 1, 80)
            
            # e_mel = torch.cat((e_mel, e_end_token), 1)
            e_mel_inp = e_start_token
            for i in range(1, len(e_mel_mask[1])):
              e_mel_postpred, e_mel_pred, e_end_logits_pred, e_attn_dec, e_attn_probs, e_attns_enc = model(e_vid, e_mel_inp)
              e_mel_inp = torch.cat((e_mel_inp, e_mel_postpred[:, -1:, :]), dim=1)

            # ======================================================================================================================

            # # for i in range(len(e_mel_mask[1])):
            # #     if i == 0:
            # e_mel_inp = torch.cat((e_start_token, empty_token), 1) # -5.,,,,-4.,,,,,, #B, 241, 80
                    
            # e_mel_postpred, e_mel_pred, e_end_logits_pred, e_attn_dec, e_attn_probs, e_attns_enc = model(e_vid, e_mel_inp)
                
            #     # e_mel_inp = e_mel_pred

            # # First loop of eval

            # e_mel_inp = torch.cat((e_start_token, empty_token), 1) # -5.,,,,-4.,,,,,, #B, 241, 80
            # e_mel_postpred, e_mel_pred, e_end_logits_pred, e_attn_dec, e_attn_probs, e_attns_enc = model(e_vid, e_mel_inp)
            # next_pred = e_mel_pred[:, 0:1]

            # # Subsequent loops
            # c=0
            # for i in range(1, len(e_mel_mask[1])+1):
            #     e_mel_inp = torch.cat((e_start_token, next_pred, empty_token[:, i:]), 1)
            #     e_mel_postpred, e_mel_pred, e_end_logits_pred, e_attn_dec, e_attn_probs, e_attns_enc = model(e_vid, e_mel_inp)
            #     next_pred = torch.cat((next_pred, e_mel_pred[:, i:i+1]), 1)
            #     c+=1
            #     if torch.equal(e_mel_pred[:, i:i+1], e_end_token) == True:
            #         # print("found end token")
            #         break
            # # print("count=", c)
            # # If prediction len is <240, fill it up with empty token
            # # print(next_pred.shape)
            # if next_pred.shape[1] != 241:
            #     # print("adding_emptytoken", next_pred.shape)
            #     x = next_pred.shape[1] # eg: shape = 200
            #     next_pred = torch.cat((next_pred, empty_token[:, 240-x:]))

            # e_mel_postnet = postnet(next_pred) + next_pred
         
           
            e_loss = loss_fn(
                e_mel,
                e_mel_postpred,
                e_mel_pred,
            )
            e_running_loss += e_loss.item()

            eval_loss.append(e_loss.detach().cpu())
            writer.add_scalars('evaluating_loss', {
                'eval_loss': e_loss,
                'eval_running_loss': e_running_loss,

            }, epoch)

            e_count += 1
            if e_count == 1:
                break
        # print("len atten dec", len(e_attn_dec))
        # attention_heads_dec = list()
        # for id, attentionheads in enumerate(e_attn_dec):
        #     attention_heads_dec.append(attentionheads.contiguous().view(-1, hparams.batch_size, 4, attentionheads.shape[1], attentionheads.shape[2])[0])
        # for i, attention_heads in enumerate(attention_heads_dec):
        #     # print("attention heads", attention_heads.shape)
        #     attn_maps = attention_heads[0]
        #     attn_maps = attn_maps.detach().cpu()
        #     for j in range(4):
        #         ax[i][j].imshow(attn_maps[j], origin='lower', vmin=0)
        #         ax[i][j].set_title("Layer %i, Head %i" % (i + 1, j + 1))
        #
        #         # print("atten heads[head]", attention_heads[j].shape)
        #         # ax = plt.gca()
        #         # ax.matshow(attention_heads[j])
        #     fig.subplots_adjust(hspace=0.5)
        #     plt.show()

            # plt.savefig('D:\\Final-sem-project\\Transformer-LST\\logs\\plots\\{}_epoch_decoder_eval_alignments_layer.png'.format(epoch))



        for i, prob in enumerate(e_attn_probs):  # j values: 0, 1, 2, 3 ; batchsize: 2 ; prob[]: 0, 2, 4, 6
            num_h = prob.size(0)
            # print("attn_prob", prob.shape)
            for j in range(4):  # Since no. of heads in decoder attention is 4
                x = vutils.make_grid(prob[j * 4] * 255)
                writer.add_image('Eval Attention_%d_0' % epoch, x, i * hparams.nhead + j)


        for i, prob in enumerate(e_attns_enc):
            # print("attn enc", prob.shape)
            num_h = prob.size(0)
            for j in range(4):
                x = vutils.make_grid(prob[j] * 255)
                writer.add_image('Eval Attention_enc_%d_0' % epoch, x, i * hparams.nhead + j)

        for i, prob in enumerate(e_attn_dec):
            num_h = prob.size(0)
            # print("attn_dec", prob.shape)
            for j in range(4):  # Since no. of heads in decoder attention is 4
                # ax[i][j].imshow(prob[j * 4].permute(1, 2, 0).detach().cpu(), origin='lower', vmin=0)
                x = vutils.make_grid(prob[j * 4] * 255)
                y = vutils.make_grid(prob[j * 4])
                writer.add_image('Eval Attention_dec_%d_0' % epoch, x, i * hparams.nhead + j)
        # y = y.permute(1, 2, 0).detach().cpu()
        # plt.imshow(y)
        # plt.savefig(
        #     'D:\\Final-sem-project\\Transformer-LST\\logs\\plots\\{}_epoch_decoder_eval_alignments.png'.format(
        #         epoch))

        writer.flush()
        e_epoch_loss = e_running_loss

        e_target = np.asarray(e_mel[0].permute(1, 0).detach().cpu())
        e_postnet_pred = np.asarray(e_mel_postpred[0].permute(1, 0).detach().cpu())
        e_pred = np.asarray(e_mel_pred[0].permute(1, 0).detach().cpu())
        plot.plot_spectrogram(e_postnet_pred,
                              os.path.join(hparams.mel_path, "step-{}-eval-mel-spectrogram.png".format(epoch)),
                              title="{}, epoch={}, loss={:.5f}".format("Transformer",
                                                                       epoch, e_epoch_loss),
                              target_spectrogram=e_target,
                              max_len=None)

        plot.plot_spectrogram(e_pred,
                              os.path.join(hparams.mel_path,
                                           "step-{}-eval-decoder_output-mel-spectrogram.png".format(epoch)),
                              title="{}, epoch={}, loss={:.5f}".format("Transformer",
                                                                       epoch, e_epoch_loss),
                              target_spectrogram=e_target,
                              max_len=None)
        target_wav = audio.inv_mel_spectrogram(e_target)
        wav = audio.inv_mel_spectrogram(e_postnet_pred)
        audio.save_wav(target_wav,
                       os.path.join(hparams.wav_path, "step-{}-target-wave-from-eval-mel.wav".format(epoch)),
                       sr=hparams.sample_rate)
        audio.save_wav(wav,
                       os.path.join(hparams.wav_path, "step-{}-wave-from-eval-mel.wav".format(epoch)),
                       sr=hparams.sample_rate)

        return e_epoch_loss, eval_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=bool, help='Global step to restore checkpoint', default=False)
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=3000)
    args = parser.parse_args()
    print("Reading data parameters...")
    model = LSTransformer()
    model = model.cuda()
    now = datetime.now()
    num = len(next(os.walk('./runs'))[1]) + 1
    writer = SummaryWriter("./runs/{}_{}".format(now.strftime("%Y-%m-%d_%H-%M-%S"), num))
    # net = nn.DataParallel(model).cuda()
    torch.manual_seed(hparams.random_seed)
    torch.cuda.manual_seed_all(hparams.random_seed)

    dataset = MyDataset(hparams.data_root, 'train')
    transforms = [
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    ]
    optimizer = transformers.AdamW(model.parameters(), lr=hparams.lr)
    
    epoch_start = 1
    model_train_loss = []
    model_eval_loss = []
    prev_flat_train_loss = []
    prev_flat_eval_loss = []

    
    model_encoder_alpha = []
    model_decoder_alpha = []
    prev_flat_encoder_alpha = []
    prev_flat_decoder_alpha = []

    if args.restore:
        print("Restoring Checkpoint...")
        # model.load_state_dict(load_checkpoint("transformer", args.restore_step))
        state_dict = torch.load('./logs/checkpoint/checkpoint_%s_%d.pth.tar' % ("transformer", args.restore_step), map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epoch_start = state_dict['epoch']
        
        data = np.load('./logs/loss/loss_epoch_{}.npz'.format(epoch_start))
        prev_flat_train_loss = data['trainloss']
        prev_flat_eval_loss = data['evalloss']
        
        # data = np.load('./logs/alpha/alpha_epoch_{}.npz'.format(epoch_start))
        # prev_flat_encoder_alpha = data['encalpha']
        # prev_flat_decoder_alpha = data['decalpha']
        model = model.cuda()
        epoch_start +=1
        

    loader = load_data_to_dataloader(dataset)

    print('num_train_data:{}'.format(len(dataset.data)))
    # print("len(dataset)", len(dataset))
    num_training_steps = hparams.epochs * len(dataset) // hparams.batch_size

    best_loss = 1e10
    if args.restore:
        print(f'---------[INFO] Restarting Training from Epoch {epoch_start} -----------\n')
    
    print('--------- [INFO] STARTING TRAINING ---------\n')
    for epoch in range(epoch_start, hparams.epochs + 1):
        train_loss, training_loss, encoder_alpha, decoder_alpha = train(model, loader, optimizer, writer, epoch, len(dataset))
        val_loss, eval_loss = evaluate(model, loader, optimizer, writer, epoch, len(dataset))
        model_train_loss.append(training_loss)
        model_eval_loss.append(eval_loss)
        model_encoder_alpha.append(encoder_alpha)
        model_decoder_alpha.append(decoder_alpha)
        print(f'EPOCH -> {epoch}/{hparams.epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss} | LR = {hparams.lr} \n')

        
        # Save checkpoint
        if epoch % hparams.save_step == 0:
            print("Saving Loss...")
            flat_train_loss = [item for sublist in model_train_loss for item in sublist]
            flat_eval_loss = [item for sublist in model_eval_loss for item in sublist]
            if len(prev_flat_train_loss) == 0:
                new_train_loss = prev_flat_train_loss + flat_train_loss
                new_eval_loss = prev_flat_eval_loss + flat_eval_loss
            else:
                new_train_loss = np.concatenate((prev_flat_train_loss, flat_train_loss), axis = None)
                new_eval_loss = np.concatenate((prev_flat_eval_loss, flat_eval_loss), axis = None)
            # print("flat train loss", len(flat_train_loss))
            # print(flat_train_loss)
            # # print("flat eval loss", len(flat_eval_loss))
            # print(flat_eval_loss)
            np.savez('./logs/loss/loss_epoch_{}'.format(epoch), trainloss=new_train_loss, evalloss=new_eval_loss)

            print("Saving Alpha...")
            flat_encoder_alpha = [item for sublist in model_encoder_alpha for item in sublist]
            flat_decoder_alpha = [item for sublist in model_decoder_alpha for item in sublist]
            if len(prev_flat_encoder_alpha) == 0:
                new_encoder_alpha = prev_flat_encoder_alpha + flat_encoder_alpha
                new_decoder_alpha = prev_flat_decoder_alpha + flat_decoder_alpha
            else:
                new_encoder_alpha = np.concatenate((prev_flat_encoder_alpha, flat_encoder_alpha), axis = None)
                new_decoder_alpha = np.concatenate((prev_flat_decoder_alpha, flat_decoder_alpha), axis = None)
            np.savez('./logs/alpha/alpha_epoch_{}'.format(epoch), encalpha=new_encoder_alpha, decalpha=new_decoder_alpha)

            arr_length = [i for i in range(len(new_train_loss))]
            # print("arr length", arr_length)
            # print("training loss", flat_train_loss)
            plt.plot(arr_length, new_train_loss, label = 'Training Loss')

            eval_arr_length = [i*10 for i in range(1, len(new_eval_loss)+1)]
            # print("eval arr length", eval_arr_length)
            plt.plot(eval_arr_length, new_eval_loss, label = 'Eval Loss')

            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./logs/loss/plots/epoch_{}_loss.png'.format(epoch), format="png")
            plt.close()
     
            print("Saving Checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(hparams.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % epoch))


