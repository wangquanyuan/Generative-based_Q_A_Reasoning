import tensorflow as tf
import numpy as np
from seq2seq_transformer_pgn_tf2.batcher import output_to_words
from seq2seq_transformer_pgn_tf2.utils.decoding import calc_final_dist
from seq2seq_transformer_pgn_tf2.layers.transformer import create_masks
from tqdm import tqdm
import math


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs

    def extend(self, token, log_prob):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob] # we add the log prob of the decoded token
                          )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params):

    def decode_onestep(enc_input, dec_input, embed_x, params,batch_oov_len,enc_extended_input):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_input, dec_input)
        embed_dec = model.embedding(dec_input)
        enc_output = model.encoder(embed_x, params["training"], enc_padding_mask)

        # dec_output.shape == (batch_size, seq_len, d_model)
        dec_output, attention_weights, p_gens = model.decoder(embed_dec,
                                                     enc_output,
                                                     params["training"],
                                                     combined_mask,
                                                     dec_padding_mask)
        #print('dec_output is ', dec_output)
        final_output = model.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.nn.softmax(final_output)
        # print('final_output is ', final_output)
        # p_gens = tf.keras.layers.Dense(tf.concat([before_dec, dec, attn_dists[-1]], axis=-1),units=1,activation=tf.sigmoid,trainable=training,use_bias=False)
        attn_dists = attention_weights['decoder_layer{}_block2'.format(model.params["num_blocks"])] # (batch_size,num_heads, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1)/model.params["num_heads"] # (batch_size, targ_seq_len, inp_seq_len)
        # print('attn_dists is ', attn_dists)
        final_dists = calc_final_dist(enc_extended_input,
                                      tf.unstack(final_output, axis=1),
                                      tf.unstack(attn_dists, axis=1),
                                      tf.unstack(p_gens, axis=1),
                                      batch_oov_len,
                                      model.params["vocab_size"],
                                      model.params["batch_size"])

        # final_dists shape=(3, 1, 30000)
        # top_k_probs shape=(3, 6)
        # top_k_ids shape=(3, 6)
        #print("------------------------------------------------------")
        #print(len(final_dists))
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists[-1]), k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)
        # dec_hidden shape = (3, 256)
        # attentions, shape = (3, 115)
        # p_gens shape = (3, 1)
        # coverages,shape = (3, 115, 1)
        results = {"top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   }
        return results

    # end of the nested class

    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    # enc_input = batch[0]["enc_input"]
    # Initial Hypothesises (beam_size many list)
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       log_probs=[0.0]
                       ) for _ in range(params['batch_size'])]
    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step
    enc_input = batch[0]["enc_input"]
    enc_extended_input = batch[0]["extended_enc_input"]
    batch_oov_len = batch[0]["max_oov_len"]
    embed_x = model.embedding(enc_input)
    dec_i = None
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # we replace all the oov is by the unknown token
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # we collect the last states for each hypothesis
        # states = [h.state for h in hyps]
        # prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        # prev_coverage = tf.convert_to_tensor(prev_coverage)

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(3, 1)
        if dec_i is None:
            dec_input = dec_input
        else:
            dec_input = tf.concat((dec_i, dec_input), 1)
        returns = decode_onestep(enc_input, dec_input, embed_x, params,batch_oov_len,enc_extended_input)
        top_k_ids, topk_log_probs = returns['top_k_ids'],returns['top_k_log_probs']
        dec_i = dec_input
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        num = 1
        for i in range(num_orig_hyps):
            h = hyps[i]
            num += 1
            for j in range(params['beam_size'] * 2):
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                #print(top_k_ids[i, j].numpy())
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j]
                                   )
                all_hyps.append(new_hyp)
        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == vocab.word_to_id('[STOP]'):
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break
        steps += 1

    if len(results) == 0:
        results = hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence,
    # given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    best_hyp = result_index2text(best_hyp, vocab, batch)
    # best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, batch[0]["article_oovs"][0])[1:-1])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()
    return best_hyp


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != 2 and index != 3:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp