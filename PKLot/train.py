from PKLot.input_data import *
from PKLot.input_data import *
from PKLot.model import *
import tensorflow as tf


def run_training():

    tfrecords_path = '/home/liming/Documents/datasets/PKLot/tfrecords'
    train_tfrecords_name = 'train'
    val_tfrecords_name = 'val'
    logs_train_dir = '/home/liming/Documents/datasets/PKLot/logs/train/'
    logs_val_dir = '/home/liming/Documents/datasets/PKLot/logs/val/'
    N_CLASSES = 2
    BATCH_SIZE = 128
    MAX_STEP = 2000
    learning_rate = 0.0001
    IMG_W = 28
    IMG_H = 28
    CHANNEL = 1

    # read data from tfrecords file
    train_batch, train_label_batch = read_and_decode(os.path.join(tfrecords_path,train_tfrecords_name + '.tfrecords'),BATCH_SIZE)
    val_batch, val_label_batch = read_and_decode(os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'),BATCH_SIZE)

    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = losses(train_logits, train_label_batch)
    train_op = training(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMG_W, IMG_H, CHANNEL])
    y_ = tf.placeholder(tf.int16, shape = [BATCH_SIZE])



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.group(tf.global_variables_initializer(),
                          # tf.local_variables_initializer()))
        # sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # starts to train
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                train_images, train_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                feed_dict={x:train_images, y_:train_labels})

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([train_loss, train_acc], feed_dict={x:val_images, y_:val_labels})
                    print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **' %(step, val_loss, val_acc*100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    run_training()