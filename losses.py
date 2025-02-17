import tensorflow as tf


def cox_loss(y_true, y_pred):
    time_value = tf.squeeze(tf.gather(y_true, [0], axis=1))
    event = tf.cast(tf.squeeze(tf.gather(y_true, [1], axis=1)), tf.bool)
    score = tf.squeeze(y_pred, axis=1)

    ix = tf.where(event)

    sel_mat = tf.cast(tf.gather(time_value, ix) <= time_value, tf.float32)

    p_lik = tf.gather(score, ix) - tf.math.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))

    loss = -tf.reduce_mean(p_lik)

    return loss


def concordance_index(y_true, y_pred):
    time_value = tf.squeeze(tf.gather(y_true, [0], axis=1))
    event = tf.cast(tf.squeeze(tf.gather(y_true, [1], axis=1)), tf.bool)
    ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
    ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                 tf.expand_dims(event, axis=-1)), name='ix')

    ## count how many score[i]<score[j]
    s1 = tf.gather(y_pred, ix[:, 0])
    s2 = tf.gather(y_pred, ix[:, 1])
    ci = tf.reduce_mean(tf.cast(s1 < s2, tf.float32), name='c_index')

    return ci



#*******************************************************************************
# MIT License
#
# Copyright (c) 2022 Kather Lab at EKFZ / TU Dresden
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so
#*********************************************************************************