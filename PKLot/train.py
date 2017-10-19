from PKLot.input_data import *
from PKLot.input_data import *
from PKLot.model import *
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_training():

    BATCH_SIZE = 128
    tfrecord_path = 'F:\\datasets\\PKLot\\tfrecords'
    tfrecord_name = 'PKLot_segmented2'
    logs_train_dir = 'F:\\python\\ML\\PKLot\\logs\\'
    MAX_STEP = 10000
    learning_rate = 0.0001

    train_batch, train_label_batch = read_and_decode(os.path.join(tfrecord_path,tfrecord_name + '.tfrecords'),BATCH_SIZE)
    train_logits = inference(train_batch, BATCH_SIZE, BATCH_SIZE)
    train_loss = losses(train_logits, train_label_batch)
    train_op = training(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.group(tf.global_variables_initializer(),
                          # tf.local_variables_initializer()))
        # sess.run(tf.local_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # starts to train
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

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