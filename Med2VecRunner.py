import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pickle
from Med2Vec import Med2Vec
from PredictModel import PredictModel


def load_data(x_file, d_file, y_file):
    x_seq = np.array(pickle.load(open("./Med2Vec_data/" + x_file, 'rb')))[:50000]
    d_seq = []
    if len(d_file) > 0:
        d_seq = np.array(pickle.load(open("./Med2Vec_data/" + d_file, 'rb')))[:50000]
    y_seq = []
    if len(y_file) > 0:
        y_seq = np.array(pickle.load(open("./Med2Vec_data/" + y_file, 'rb')))[:50000]
    return x_seq, d_seq, y_seq


def pickTwo(codes, iVector, jVector):
    for first in codes:
        for second in codes:
            if first == second: continue
            iVector.append(first)
            jVector.append(second)


def pad_matrix(seqs, labels, config):
    n_samples = len(seqs)
    i_vec = []
    j_vec = []
    n_input = config['n_input']
    n_output = config['n_output']

    if n_input > n_output:
        x = np.zeros((n_samples, n_input))
        y = np.zeros((n_samples, n_output))
        mask = np.zeros((n_samples, 1))
        for idx, (seq, label) in enumerate(zip(seqs, labels)):
            if not seq[0] == -1:
                x[idx][seq] = 1.
                y[idx][label] = 1.
                pickTwo(seq, i_vec, j_vec)
                mask[idx] = 1.
        return x, y, mask, i_vec, j_vec
    else:
        x = np.zeros((n_samples, n_input))
        mask = np.zeros((n_samples, 1))
        for idx, seq in enumerate(seqs):
            if not seq[0] == -1:
                x[idx][seq] = 1.
                pickTwo(seq, i_vec, j_vec)
                mask[idx] = 1.
        return x, mask, i_vec, j_vec


def precision_top(y_true, y_pred, rank=None):
    if rank is None:
        rank = [1, 2, 3, 4, 5]
    pre = list()
    for i in range(len(y_pred)):
        thisOne = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_pred[i])
            for rk in rank:
                if len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) >= 1:
                    thisOne.append(1)
                else:
                    thisOne.append(0)
            pre.append(thisOne)

    return (np.array(pre)).mean(axis=0).tolist()


def get_config():
    config = dict()
    config['init_scale'] = 0.01
    config['n_windows'] = 1
    config['n_input'] = 2457
    config['n_emb'] = 200
    config['n_demo'] = 0
    config['n_hidden'] = 200
    config['n_output'] = 748
    config['max_epoch'] = 20
    config['n_samples'] = 50000  # 350671
    config['batch_size'] = 256
    config['display_step'] = 1
    return config


def model_train(med2vec, saver, config):
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        total_batch = int(np.ceil(config['n_samples'] / config['batch_size']))
        x_seq, d_seq, y_seq = load_data('seqs.pkl', '', 'labels.pkl')
        # Loop over all batches
        for index in range(total_batch):
            print(index)
            x_batch = x_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = []
            if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)
            elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
                d_batch = d_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                x, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, d=d_batch, mask=mask, i_vec=i_vec, j_vec=j_vec)
            elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
                y_batch = y_seq[index * config['batch_size']: (index + 1) * config['batch_size']]
                x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, y=y, mask=mask, i_vec=i_vec, j_vec=j_vec)
            else:
                x, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
                cost = med2vec.partial_fit(x=x, mask=mask, i_vec=i_vec, j_vec=j_vec)
            # Compute average loss
            avg_cost += cost / config['n_samples'] * config['batch_size']
        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        if epoch == config['max_epoch'] - 1:
            saver.save(sess=med2vec.sess, save_path='./Med2Vec_model/2emb_2hidden/med2vec',
                       global_step=config['max_epoch'])


def show_code_representation(med2vec, saver):
    ckpt = tf.train.get_checkpoint_state('./Med2Vec_model/2emb_2hidden')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    w_emb, w_hidden, w_output = med2vec.get_weights()
    labels = ["GXB", "TNB", "GXYB", "NGSHYZ", "MXBDXGY", "NGS", "NCZ"]
    disease_list = [1, 2, 499, 826, 168, 169, 1175]
    plt.scatter(w_emb[disease_list, 0], w_emb[disease_list, 1], s=10)
    for label, x, y in zip(labels, w_emb[disease_list, 0], w_emb[disease_list, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.savefig('./Med2Vec_fig/code_show.png', dpi=700)


def interpret_code_representation(med2vec, saver):
    ckpt = tf.train.get_checkpoint_state('./Med2Vec_model/200emb_200hidden')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    w_emb, w_hidden, w_output = med2vec.get_weights()
    code_dict = pickle.load(open("./Med2Vec_data/code_dict.pkl", 'rb'))
    for i in code_dict:
        print(i,code_dict[i])
    for i in range(get_config()['n_emb']):
        print(i,end=' ')
        sorted_code = np.argsort(w_emb[:, i])[get_config()['n_input'] - 10:get_config()['n_input']]
        for j in sorted_code:
            print(code_dict[j], end=' ')
        print()


def predict_next_visit(med2vec, saver, config):
    ckpt = tf.train.get_checkpoint_state('./Med2Vec_model/200emb_200hidden')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(med2vec.sess, ckpt.model_checkpoint_path)
    x_seq, d_seq, y_seq = load_data('seqs.pkl', '', 'labels.pkl')
    if config['n_demo'] > 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] > 0 and config['n_input'] == config['n_output']:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    elif config['n_demo'] == 0 and config['n_input'] > config['n_output']:
        x, y, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)
    else:
        x, mask, i_vec, j_vec = pad_matrix(x_seq, y_seq, config)
        visit_seq = med2vec.get_visit_representation(x=x)

    x_seq, d_seq, y_seq = load_data('seqs.pkl', '', 'labels.pkl')
    x_seq_new = []
    y_seq_new = []
    visit_seq_new = []
    for i in range(config['n_samples'] - 1):
        if x_seq[i][0] != -1 and y_seq[i + 1][0] != -1:
            x_seq_new.append(x_seq[i])
            visit_seq_new.append(visit_seq[i])
            y_seq_new.append(y_seq[i - 1])
    for i in range(10):
        for j in visit_seq_new[i]:
            print(j, end=' ')
        print()

    predict_model1 = PredictModel(n_input=config['n_input'], n_output=config['n_output'])
    total_batch = int(np.ceil(len(x_seq_new) * 0.8 / config['batch_size']))
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
            cost = predict_model1.partial_fit(x=x, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size'] * 0.8

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    x, y, mask, i_vec, j_vec = pad_matrix(x_seq_new[int(0.8 * len(x_seq_new)):],
                                          y_seq_new[int(0.8 * len(x_seq_new)):], config)
    predict_y = predict_model1.get_result(x=x)
    print(precision_top(y, predict_y))

    predict_model2 = PredictModel(n_input=config['n_emb'], n_output=config['n_output'])
    total_batch = int(np.ceil(len(x_seq_new) * 0.8 / config['batch_size']))
    for epoch in range(config['max_epoch']):
        avg_cost = 0.
        for index in range(total_batch):
            x_batch = x_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            y_batch = y_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            visit_batch = visit_seq_new[index * config['batch_size']: (index + 1) * config['batch_size']]
            x, y, mask, i_vec, j_vec = pad_matrix(x_batch, y_batch, config)
            cost = predict_model2.partial_fit(x=visit_batch, y=y)
            avg_cost += cost / len(x_seq_new) * config['batch_size'] * 0.8

        if epoch % config['display_step'] == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    x, y, mask, i_vec, j_vec = pad_matrix(x_seq_new[int(0.8 * len(x_seq_new)):],
                                          y_seq_new[int(0.8 * len(x_seq_new)):], config)
    predict_y = predict_model2.get_result(x=visit_seq_new[int(0.8 * len(x_seq_new)):])
    print(precision_top(y, predict_y))


def main(_):
    config = get_config()
    med2vec = Med2Vec(n_input=config['n_input'], n_emb=config['n_emb'], n_demo=config['n_demo'],
                      n_hidden=config['n_hidden'], n_output=config['n_output'], n_windows=config['n_windows'])
    saver = tf.train.Saver()
    # model_train(med2vec, saver, config)
    # show_code_representation(med2vec, saver)
    # predict_next_visit(med2vec, saver, config)
    interpret_code_representation(med2vec, saver)


if __name__ == "__main__":
    tf.app.run()
